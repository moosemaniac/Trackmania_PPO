import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PPOMemory:
    def __init__(self, batch_size = 256):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0,n_states, self.batch_size)
        indices = np.arange (n_states, dtype=np.int64)
        np.random.shuffle (indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), np.array(self.probs), np.array(self.values), np.array(self.actions), np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self, state, probs, values, action, rewards, dones):
        

        if isinstance(state, torch.Tensor):
             state = state.squeeze(0).cpu().numpy()
        elif isinstance(state, np.ndarray):
             state = state.squeeze()
        
        if isinstance(action, torch.Tensor):
             action = action.squeeze(0).cpu().numpy()
        elif isinstance(action, np.ndarray):
             action = action.squeeze()

        if isinstance(probs, torch.Tensor):
             probs = probs.squeeze(0).cpu().numpy()
        elif isinstance(probs, np.ndarray):
             probs = probs.squeeze(axis=0)

        if isinstance(values, torch.Tensor):
             values = values.item()
             
        self.states.append(state)
        self.probs.append(probs)
        self.values.append(values)
        self.actions.append(action)
        self.rewards.append(rewards)
        self.dones.append(dones)
        
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        

LOG_STD_MAX = 2
LOG_STD_MIN = -20
        
class ActorNetwork (nn.Module):
    def __init__ (self, action_dim, input_dims, alpha, fc1_dims = 256, fc2_dims = 256, chkpt_dir = "tmp/ppo"):
        super(ActorNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_ppo")
        # Feature layers
        self.actor_features = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU()
        )
        
        # Mean for the action distribution
        self.actor_mean = nn.Linear(fc2_dims, action_dim)
        
        # Log standard deviation for the action distribution
        # Initialize to a small negative value for small initial std
        self.actor_log_std = nn.Linear(fc2_dims, action_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.to(self.device)
      
    def forward(self, state):
        features = self.actor_features(state)
        mean = self.actor_mean(features)
        # Tanh squashing for bounded actions
        mean = torch.tanh(mean)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        # Create normal distribution with learned mean and std
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        
        return dist
        
    
    def save_checkpoint (self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims = 256, fc2_dims = 256, chkpt_dir = 'tmp/ppo'):
        super(CriticNetwork,self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')
        self.critic = nn.Sequential (
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        
        self.optimizer = optim.Adam ( self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.to(self.device)
        
    def forward(self,state):
        value = self.critic(state)
        
        return value
    
    def save_checkpoint (self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    

class Agent:
    def __init__(self,n_actions, input_dims, gamma = 0.99, alpha = 0.0003, policy_clip = 0.2, batch_size = 64, N = 2048, n_epochs = 10, gae_lambda = 0.95):
        
        self.gamma = gamma
        self.policy_clip = policy_clip
        self. alpha = alpha
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        
        self.actor = ActorNetwork(n_actions,input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember (self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state,probs,vals,action,reward,done)  
        
    def save_models(self):
        print('SAVING MODELS...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_models(self):
        print('LOADING MODELS...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
    
    def choose_action(self, observation):
        flattened_observation = np.concatenate([np.array(o).flatten() for o in observation])
        state = torch.from_numpy(flattened_observation).float().to(self.actor.device)
        state = state.unsqueeze(0)

        with torch.no_grad():
            dist = self.actor(state)
            value = self.critic (state)
            action = dist.sample()
        
        probs = dist.log_prob(action)
        action = action.cpu().numpy().flatten()
        # Explicitly clamp actions to [-1, 1] range for Trackmania control
        action = np.clip(action, -1.0, 1.0)
        value = torch.squeeze(value).item()
        
        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, old_probs_arr, vals_arr, action_arr, reward_array, done_arr, batches = self.memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros(len(reward_array),dtype=np.float32)
            #Calculate advantage
            for t in range(len(reward_array)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_array)-1):
                    v_k = values[k]
                    v_k_plus_1 = values[k+1]
                    if hasattr(v_k, '__len__') and len(v_k) > 1:
                        v_k = v_k[0]  
                    if hasattr(v_k_plus_1, '__len__') and len(v_k_plus_1) > 1:
                        v_k_plus_1 = v_k_plus_1[0] 
                    a_t += discount * (reward_array[k] + self.gamma * v_k_plus_1 * (1-int(done_arr[k])) - v_k)
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
                
            advantage = torch.tensor(advantage).to(self.actor.device)
            
            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = torch.tensor (action_arr[batch]).to(self.actor.device)
                
                #get new probability distribution
                dist = self.actor(states)
                critic_value = self.critic (states)
                
                critic_value = torch.squeeze(critic_value)
                
                new_probs = dist.log_prob(actions)
                prob_ratio = torch.exp (new_probs - old_probs)
                
                # For policy gradient calculation in continuous space, we average across action dimensions
                prob_ratio_sum = prob_ratio.sum(dim=1)
                weighted_probs = advantage [batch] * prob_ratio_sum
                weighted_clipped_probs = torch.clamp (prob_ratio_sum, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_clipped_probs, weighted_probs).mean()
                
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()
                
                total_loss=  actor_loss +0.5*critic_loss
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()
                
                
                
# Consistent observation processing functions
def process_observation(observation):
    """Process an observation into a flattened tensor"""
    if isinstance(observation, torch.Tensor):
        # If already a tensor, ensure it's flattened
        return observation.view(-1)
    elif isinstance(observation, np.ndarray):
        # If numpy array, flatten and convert to tensor
        return torch.tensor(observation.flatten())
    elif isinstance(observation, list) or isinstance(observation, tuple):
        # If it's a list of observations (like from the environment)
        flattened_observation = np.concatenate([np.array(o).flatten() for o in observation])
        return torch.tensor(flattened_observation)
    else:
        raise ValueError(f"Unsupported observation type: {type(observation)}")     
    
                
import tmrl 
import numpy as np
  
  
if __name__ == '__main__':
    # Create the Trackmania environment
    env = tmrl.get_environment()
    
    # Calculate observation space size
    observation_space = np.sum([np.prod(value.shape) for value in env.observation_space])
    
    # Get action space dimensions
    action_space = env.action_space.shape[0]
    
    # Make sure checkpoint directory exists
    os.makedirs("tmp/ppo", exist_ok=True)
    
    # Hyperparameters
    N = 2048  # Steps before update
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    
    # Create agent
    agent = Agent(n_actions=action_space, 
                  batch_size=batch_size,
                  alpha=alpha,
                  n_epochs=n_epochs, 
                  input_dims=int(observation_space))  
    
    n_episodes = 300
    
    best_score = 0
    score_history = []
    
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    for i in range(n_episodes):
        # Get the initial observation from the environment
        observation = env.reset()[0]
        
        # Process the observation to ensure consistent format
        observation = process_observation(observation)
        
        done = False
        score = 0
        
        while not done: 
            # Choose action based on current observation
            action, prob, val = agent.choose_action(observation)
            
            # Take a step in the environment
            step_result = env.step(action)
            observation_, reward, terminated, truncated, info = step_result
            
            # Process the new observation
            observation_ = process_observation(observation_)
            
            done = terminated or truncated
            
            n_steps += 1
            score += reward 
            
            # Store the transition in memory
            agent.remember(observation, action, prob, val, reward, done)
            
            # Learn if it's time
            if n_steps % N == 0:
                print(f"Learning after {n_steps} steps...")
                agent.learn()
                learn_iters += 1
                
            observation = observation_
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print("episode", i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
        