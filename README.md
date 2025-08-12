# 🏎️ Autonomous Racing Agent using Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Training Hours](https://img.shields.io/badge/Training%20Hours-220+-green.svg)]()
[![Episodes](https://img.shields.io/badge/Episodes-5600+-brightgreen.svg)]()

> **Deep Reinforcement Learning implementation for autonomous vehicle navigation in dynamic racing environments**

📖 **[Versión en Español](README_ES.md)** | **[English Version](README.md)**

### 🌍 Project Context
This project was developed as a Bachelor's Degree at **Universitat Oberta de Catalunya (UOC), Spain**. The implementation and code are written in Spanish, while this documentation is provided in English for broader international accessibility and professional presentation.


## 🚀 Project Overview

This project implements a **Proximal Policy Optimization (PPO) agent** trained from scratch to navigate racing circuits autonomously in Trackmania 2020. The agent learns to make real-time driving decisions using LIDAR sensor data, achieving significant performance improvements through deep reinforcement learning.

### 🎯 Key Achievements
- **📈 625% performance improvement** over 5,600+ training episodes 
- **🧠 Custom PPO implementation** built from ground up
- **📊 Advanced policy optimization** using actor-critic architecture
- **🏁 Autonomous navigation** in complex racing environments
- **🔬 Rigorous experimental methodology** with statistical validation

## 🛠️ Technical Architecture

### State Space Design (83 dimensions)
- **LIDAR Sensors**: 4 temporal frames × 19 rays = 76 distance measurements
- **Vehicle Dynamics**: 1 velocity feature
- **Action History**: 6 features (2 previous actions × 3 dimensions)

### Action Space (Continuous Control)
- **Acceleration/Braking**: [-1, 1] (forward/reverse throttle)
- **Steering Control**: [-1, 1] (left/right wheel angle)
- **Auxiliary Control**: [-1, 1] (additional vehicle control)

## 📊 Performance Metrics

| Training Phase | Average Reward | Episode Length | Success Rate | Key Improvements |
|---------------|----------------|----------------|--------------|------------------|
| **Initial** | 12.23 | ~500 steps | <15% | Random exploration |
| **Mid-Training** | 17.83 | ~1000 steps | ~40% | Basic navigation |
| **Final** | 87.17 | 2500 steps | >85% | **Expert racing** |
| **Improvement** | **+625%** | **+400%** | **+467%** | **Converged policy** |

### Training Statistics
- **Total Episodes**: 5,608
- **Total Timesteps**: 4,529,231
- **Training Duration**: 220 hours, 1 minute, 22 seconds
- **Peak Performance**: 111.35 reward (Episode 5503)
- **Convergence**: Stable performance in final 200 episodes

### Academic Validation
**Master's Thesis**: "Entrenamiento de un agente de aprendizaje por refuerzo en Trackmania"
- **Institution**: Universitat Oberta de Catalunya (UOC)
- **Year**: 2025
- **Program**: Computer Science Engineering - Artificial Intelligence Specialization
- **Supervisor**: Gabriel Moyà Alcover
- **Thesis Grade**: [9.3]

### Industry Relevance
This project demonstrates skills directly applicable to:
- **Autonomous Vehicles**: Real-time decision making and path planning
- **Robotics**: Continuous control and sensor integration
- **Game AI**: Intelligent agent behavior and real-time systems
- **Machine Learning Research**: Advanced RL algorithm implementation

## 📚 Technologies & Skills Demonstrated

### Core AI/ML Technologies
- **Deep Reinforcement Learning** (PPO, Actor-Critic, Policy Gradients)
- **Neural Network Architecture Design** (PyTorch, Custom Implementations)
- **Hyperparameter Optimization** (Systematic tuning, Performance analysis)
- **Computer Vision** (LIDAR processing, State representation)

### Software Engineering Excellence
- **Object-Oriented Programming** (Clean architecture, Design patterns)
- **Performance Optimization** (GPU acceleration, Efficient algorithms)
- **Real-Time Systems** (Low-latency decision making, System integration)

### Research & Analysis Skills
- **Experimental Design** (Hypothesis testing, Statistical validation)
- **Data Analysis & Visualization** (Matplotlib, Statistical metrics)
- **Technical Writing** (Academic thesis, Clear documentation)
- **Problem Solving** (Algorithm debugging, Performance optimization)

## 📖 References & Resources

### Key Academic Papers
- Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- Haarnoja et al. (2018) - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Sutton & Barto (2018) - "Reinforcement Learning: An Introduction"
- Lillicrap et al. (2015) - "Continuous Control with Deep Reinforcement Learning"

### Technical Resources
- **TMRL Framework**: Trackmania Reinforcement Learning community project
- **OpenPlanet**: Trackmania game integration middleware
- **PyTorch Documentation**: Deep learning framework reference
- **Gymnasium**: Reinforcement learning environment standard


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for full details.

The MIT License allows for both academic and commercial use, modification, and distribution.

## 👨‍💻 Author

**Carlos Expósito Carrera**
- 🎓 M.Sc. Computer Science, Artificial Intelligence Specialization
- 🏫 Universitat Oberta de Catalunya (UOC)
- 🔬 Research Focus: Deep Reinforcement Learning, Autonomous Systems
- 📧 Email: [cexposito1@gmail.com](mailto:cexposito1@gmail.com)
- 💼 LinkedIn: [Carlos Expósito Carrera](https://www.linkedin.com/in/carlos-exposito-carrera/)
- 🐙 GitHub: [@moosemaniac](https://github.com/moosemaniac)
