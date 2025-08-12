import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tmrl 
from torch.distributions import MultivariateNormal
import time
import random
import matplotlib.pyplot as plt
import csv
from pathlib import Path

    
def env_obs_to_tensor(obs, device):
    obs = [obs]  
    
    batch_tensors = []
    
    for observation in obs:
        speed, lidar_data, last_action, second_last_action = observation
        speed_array = np.array(speed / 1000)
        speed_tensor = torch.tensor(speed_array, device=device).view(-1)        
        lidar_tensor = torch.tensor(lidar_data / 400, device=device).view(-1)
        last_action_tensor = torch.tensor(last_action, device=device).view(-1)
        second_last_action_tensor = torch.tensor(second_last_action, device=device).view(-1)
        
        # Concatenate for this observation
        observation_tensor = torch.cat((
            speed_tensor, 
            lidar_tensor, 
            last_action_tensor, 
            second_last_action_tensor
        ), dim=0).float()
        
        batch_tensors.append(observation_tensor)
    
    # Stack all observations in the batch
    return torch.stack(batch_tensors).to(device)
    

class ActorNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer=512):
        super(ActorNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nn = nn.Sequential(
            nn.Linear(in_dim, n_layer),
            nn.ReLU(),
            nn.Linear(n_layer, n_layer),
            nn.ReLU()
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(n_layer, out_dim),
            nn.Tanh()
        )
        
        self.log_std_layer = nn.Linear(n_layer, out_dim)
        
        self.to(self.device)
    
    def forward(self, obs):
        # Convert obs to tensor 
        obs = env_obs_to_tensor(obs, self.device) 
        features = self.nn(obs)
        
        mean = self.mean_layer(features)
        
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20,2)  # Reasonable bounds (inspired by TMRL tutorial in SAC)
        
        return mean, log_std


class CriticNetwork(nn.Module):
    def __init__(self, in_dim, n_layer=512):
        super(CriticNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        
        self.nn = nn.Sequential(
            nn.Linear(in_dim, n_layer),
            nn.ReLU(),
            nn.Linear(n_layer, n_layer),
            nn.ReLU(),
            nn.Linear(n_layer, 1)
        )
        self.to(self.device)
    
    def forward(self, obs):
        # Convert obs to tensor 
        obs = env_obs_to_tensor(obs, self.device) 
        value = self.nn(obs)
        return value

class PPO:
    def __init__(self):
        self.init_hyperparams()
        self.env = tmrl.get_environment()
        self.obs_dim = np.sum([np.prod(value.shape) for value in self.env.observation_space])
        self.act_dim = self.env.action_space.shape[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        # Create actor and critic networks
        self.actor = ActorNetwork(self.obs_dim, self.act_dim, n_layer=512)
        self.critic = CriticNetwork(self.obs_dim, n_layer=512)

        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],    # losses of critic network in current iteration
            'total_rewards': [],    # total rewards per episode for plotting
        }
        self.init_csv_logger()
              
    def init_hyperparams(self):
        """Initialize hyperparameters for the PPO algorithm"""
        self.mini_batch_size = 256  # Batch size for training
        self.max_timesteps_per_episode = 2500
        self.gamma = 0.99  # discount factor
        self.epochs = 8    
        self.clip = 0.15    
        self.lr = 1e-4          
        self.entropy_coef = 0.015    
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            # Query the actor network for a mean action
            mean, log_std = self.actor(obs)
            
            if deterministic:
                # For evaluation, just use the mean action (This isn't working correctly, maybe too early to get good mean actions)
                action = torch.clamp(mean, -1.0, 1.0)
                return action.cpu().numpy(), torch.tensor(0.0)
            
            
            # Get standard deviations from log_std
            std = torch.exp(log_std)
            
            # Create diagonal covariance matrix
            cov_mat = torch.diag_embed(std.pow(2))
            
            # Create multivariate normal distribution
            dist = MultivariateNormal(mean, cov_mat)

            # Sample action
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action)
            
            #print(action)        
        return action.cpu().numpy(), log_prob.detach()
    
    def compute_rtgs(self, batch_rewards):
        # The rewards-to-go (rtg) per episode per batch to return
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rewards):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)
        return batch_rtgs
    
    def rollout(self):
        # Episode data
        ep_obs = []
        ep_acts = []
        ep_logprobs = []
        ep_rewards = []
        
        # Reset environment
        obs = self.env.reset()[0]
        done = False
        truncated = False
        max_wall_frames = 2
        wall_distance_threshold = 55
        wall_contact_frames = 0  
        for step in range(self.max_timesteps_per_episode):
            # Store observation
            ep_obs.append(obs)
            
            # Get action and log probability
            action, logprob = self.get_action(obs)
            
            # Step environment
            obs, reward, done, truncated, info = self.env.step(action[0])
            
            '''# Wall detection from LIDAR data
            min_distance = obs[1].min() if isinstance(obs[1], np.ndarray) else float('inf')
            near_wall = min_distance < wall_distance_threshold
            
            if near_wall:
                wall_contact_frames += 1
            else:
                wall_contact_frames = 0
            if wall_contact_frames >= max_wall_frames:
                done = True'''
            done = done or truncated
                
            # Store action, logprob, and reward
            ep_acts.append(action)
            ep_logprobs.append(logprob)
            ep_rewards.append(reward)
            
            if done:
                self.env.reset()
                break
        
        # Convert to tensors
        ep_acts = torch.tensor(np.vstack(ep_acts), dtype=torch.float, device=self.device)
        ep_logprobs = torch.tensor(ep_logprobs, dtype=torch.float, device=self.device)
        
        # Calculate rewards-to-go
        ep_rtgs = self.compute_rtgs([ep_rewards])
        
        return ep_obs, ep_acts, ep_logprobs, ep_rtgs, ep_rewards
      
    def evaluate(self, batch_obs, batch_acts):
        # Get value estimates
        V = torch.zeros(len(batch_obs), device=self.device)
        for i, obs in enumerate(batch_obs):
            V[i] = self.critic(obs).squeeze()
        
        # Get log probabilities of actions taken
        log_probs = torch.zeros(len(batch_obs), device=self.device)
        entropy = 0.0
        for i, (obs, act) in enumerate(zip(batch_obs, batch_acts)):
            mean, log_std = self.actor(obs)
            std = torch.exp(log_std) 
            cov_mat = torch.diag_embed(std.pow(2))
            dist = MultivariateNormal(mean, cov_mat)
            log_probs[i] = dist.log_prob(act)
            entropy += dist.entropy().mean()
        
        return V, log_probs, entropy/len(batch_obs)
                
    def train_on_episode(self, ep_obs, ep_acts, ep_logprobs, ep_rtgs):
        """
        Train on episode data, always using mini-batches
        """
        # Calculate advantages
        V, _ , _ = self.evaluate(ep_obs, ep_acts)
        A_k = ep_rtgs - V.detach()
        # Normalize advantages
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)
        
        # Convert everything to lists for easier indexing
        indices = list(range(len(ep_obs)))
        
        actor_losses = []
        critic_losses = []
        
        # Training loop for multiple epochs
        for _ in range(self.epochs):
            # Shuffle the indices for each epoch
            random.shuffle(indices)
            epoch_actor_losses = []
            epoch_critic_losses = []
            num_batches = 0
            
            # Process mini-batches
            for i in range(0, len(indices), self.mini_batch_size):
                # Get batch indices
                batch_indices = indices[i:i + self.mini_batch_size]
                if len(batch_indices) == 0:
                    continue
                    
                num_batches += 1
                
                # Get batch data
                batch_obs = [ep_obs[idx] for idx in batch_indices]
                batch_acts = ep_acts[batch_indices]
                batch_logprobs = ep_logprobs[batch_indices]
                batch_rtgs = ep_rtgs[batch_indices]
                batch_A_k = A_k[batch_indices]
                
                # Evaluate current policy on this batch
                V_batch, current_logprobs_batch, entropy = self.evaluate(batch_obs, batch_acts)
                
                # Calculate ratios
                ratios = torch.exp(current_logprobs_batch - batch_logprobs)
                
                # Calculate surrogate losses
                surr1 = ratios * batch_A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_A_k
                # Calculate actor and critic losses
                actor_loss = (-torch.min(surr1, surr2)).mean()
                actor_loss = actor_loss -self.entropy_coef * entropy
                epoch_actor_losses.append(actor_loss.item())
                critic_loss = nn.MSELoss()(V_batch, batch_rtgs)
                epoch_critic_losses.append(critic_loss.item())
                
                # Update actor network 
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()
                
                # update critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()
                


            if num_batches > 0:
                actor_losses.append(np.mean(epoch_actor_losses))
                critic_losses.append(np.mean(epoch_critic_losses))
        
        # Return the average losses from the last epoch
        return np.mean(actor_losses), np.mean(critic_losses)
                
    def learn(self, total_episodes):
        episode_count = self.logger['i_so_far']
        
        while episode_count < total_episodes:
            # Record start time for this episode
            episode_start_time = time.time()


            # Collect one episode
            ep_obs, ep_acts, ep_logprobs, ep_rtgs, ep_rewards = self.rollout()
            
            # Update counters
            episode_count += 1
            self.logger['i_so_far'] = episode_count
            self.logger['batch_rews'].append(ep_rewards)
            self.logger['batch_lens'].append(len(ep_rewards))
            self.logger['t_so_far'] += len(ep_rewards)
            episode_length = len(ep_obs)

            
            print(f"Episode {episode_count} length: {episode_length}")
            
            # Train with the chosen method
            actor_loss, critic_loss = self.train_on_episode(ep_obs, ep_acts, ep_logprobs, ep_rtgs)
            
            # Get total episode reward
            total_reward = sum(ep_rewards)
            self.logger['total_rewards'].append(total_reward)
            # Calculate episode duration
            episode_duration = time.time() - episode_start_time
            # Log to CSV
            self.save_to_csv(
                episode=episode_count,
                total_reward=total_reward,
                episode_length=episode_length,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                time_taken=episode_duration
            )
            self.env.reset()
            # Print stats
            print(f"Episode {episode_count}")
            print(f"Total Reward: {total_reward}")
            print(f"Episode Length: {len(ep_rewards)}")
            print(f"Actor Loss: {actor_loss}")
            print(f"Critic Loss: {critic_loss}")
            print("-" * 50)
            
            # Keep regular saving every 5 episodes
            if episode_count % 5 == 0:
                print(f"Regular checkpoint at episode {episode_count}")
                torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'actor_optim': self.actor_optim.state_dict(),
                    'critic_optim': self.critic_optim.state_dict(),
                    'episode': episode_count,
                    'reward': total_reward
                }, f'checkpoint_episode_{episode_count}.pt')
                
                # Log summary 
                self._log_summary()
    
    
    def _log_summary(self):
        # Calculate logging values
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'][-5:])  # Last 5 episodes
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews'][-5:]])  # Last 5 episodes
        
        # Fix: Add .cpu() before converting to numpy
        if self.logger['actor_losses']:
            avg_actor_loss = np.mean([loss.cpu().float().mean().item() for loss in self.logger['actor_losses'][-20:]])  # Last 20 updates
        else:
            avg_actor_loss = 0
            
        if self.logger['critic_losses']:
            avg_critic_loss = np.mean([loss.cpu().float().mean().item() for loss in self.logger['critic_losses'][-20:]])  # Last 20 updates
        else:
            avg_critic_loss = 0

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 10))
        avg_critic_loss = str(round(avg_critic_loss, 10))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

    def evaluate_policy(self, num_episodes=5):
        total_rewards = []
        
        for i in range(num_episodes):
            # Reset environment
            obs = self.env.reset()[0]
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                # Get action deterministically (using mean, no sampling)
                action, _ = self.get_action(obs, deterministic=True)
                # Step environment
                obs, reward, done, truncated, info = self.env.step(action[0])
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            print(f"Evaluation Episode {i+1}: Reward = {episode_reward}")
        
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average Evaluation Reward: {avg_reward}")
        
        return avg_reward

    def load_model(self, path):
        try:
            # First try with weights_only=False (for compatibility with older PyTorch versions)
            checkpoint = torch.load(path, weights_only=False)
        except (TypeError, AttributeError):
            # Fallback for older PyTorch versions that don't have weights_only parameter
            checkpoint = torch.load(path)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"Loaded model from {path}")
        
        # Optionally load optimizer state
        if 'actor_optim' in checkpoint and 'critic_optim' in checkpoint:
            self.actor_optim.load_state_dict(checkpoint['actor_optim'])
            self.critic_optim.load_state_dict(checkpoint['critic_optim'])
            print("Loaded optimizer states")
            
        if 'episode' in checkpoint:
            print(f"Model was trained for {checkpoint['episode']} episodes")
        
        if 'reward' in checkpoint:
            print(f"Model achieved reward of {checkpoint['reward']}")



    def init_csv_logger(self, filename='training_log.csv'):
        """
        Initialize a CSV file for logging training metrics.
        If the file already exists, it will not be overwritten.
        
        """
        self.log_file = filename
        csv_path = Path(filename)
        last_episode = 0
        
        # Check if file already exists (for resuming training)
        if csv_path.exists():
            print(f"Found existing training log: {filename}")
            
            # Read the last line to get the latest episode number
            with open(filename, 'r') as csvfile:
                # Skip to the end to check if file has content
                csvfile.seek(0, os.SEEK_END)
                if csvfile.tell() > 0:
                    csvfile.seek(0)
                    # Read all lines and get the last one
                    lines = csvfile.readlines()
                    if len(lines) > 1:  # More than just the header
                        last_line = lines[-1].strip()
                        # Extract episode number from first column
                        last_episode = int(last_line.split(',')[0])
                        print(f"Resuming from episode {last_episode + 1}")
                            
        else:
            print(f"Creating new training log: {filename}")
            self._create_new_csv()
        
        return last_episode

    def _create_new_csv(self):
        """Create a new CSV file with headers"""
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'episode', 
                'total_reward', 
                'episode_length',
                'actor_loss',
                'critic_loss',
                'time_taken',
                'total_timesteps'
            ])

    def save_to_csv(self, episode, total_reward, episode_length, actor_loss, critic_loss, time_taken):
        """
        Save episode metrics to CSV
        """
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                episode,
                total_reward,
                episode_length,
                actor_loss,
                critic_loss,
                time_taken,
                self.logger['t_so_far']
            ])
            
if __name__ == '__main__':
    model = PPO()
    
    # Check for existing checkpoints
    import glob
    import re
    
    checkpoint_files = glob.glob('checkpoint_episode_*.pt')
    
    # Default starting episode if no checkpoints found
    starting_episode = 0
    
    if checkpoint_files:
        # Extract episode numbers from filenames
        episode_numbers = []
        for file in checkpoint_files:
            match = re.search(r'checkpoint_episode_(\d+)\.pt', file)
            if match:
                episode_numbers.append(int(match.group(1)))
        
        if episode_numbers:
            # Find the latest checkpoint
            latest_episode = max(episode_numbers)
            latest_checkpoint = f'checkpoint_episode_{latest_episode}.pt'
            
            print(f"Found latest checkpoint: {latest_checkpoint} (Episode {latest_episode})")
            
            # Load the latest checkpoint
            model.load_model(latest_checkpoint)
            
            # Update the starting episode count in the logger
            model.logger['i_so_far'] = latest_episode
            
            # Set the starting episode for our next round
            starting_episode = latest_episode
        else:
            print("No valid checkpoints found. Starting training from scratch...")
    else:
        print("No checkpoints found. Starting training from scratch...")
    
    # Train for another 100 episodes from where we left off
    print(f"Starting training from episode {starting_episode} for more episodes...")
    
    # Update the total episodes target
    total_episodes = starting_episode + 5000
    
    # Train the model for 100 more episodes
    #model.learn(total_episodes
    
    # Evaluate the trained policy
    model.evaluate_policy(5)