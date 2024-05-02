# EE675: Course Project
This project is a near implementation of the research paper titled, "Mobile Energy Transmitter Scheduling in Energy Harvesting IoT Networks using Deep Reinforcement Learning".

Here we have implemented the environment and reward function as suggested by the above mentioned paper.

### Requirements:
```
python --version - 3.6.9 (min)
# If python2 available by default, use pip3 instead of pip
```
### Installation :
All required libraries with versions are entered in the setup.py
```bash
git clone git@github.com:rrustagi20/rl_iot.git
cd rl_iot
python3 -m venv .env
source .env/bin/activate
pip3 install -e .
```
### Executing the RL Code:

There are 2 agents implemented in the module
1. PPO (Proximal Policy Optimisation) Algorithm
2. DDPG (Deep Deterministic Policy Gradient) Algorithm

To train the PPO algorithm:
```bash
$ python3 ppo_main.py
```
To train the DDPG algorithm:
```bash
$ python3 ddpg_main.py
```
### 2 Methods are used in this project
a. Above mentioned paper MDP approach: Coupled Reward Optimisation Problem

b. A Novel Approach: Decoupled Reward System and external target guidance

To visualise the already trained rewards:

We are using tensorboard to visualise logs of the training agent. 
The below command is run parallely in another terminal while the agent is training.
After running this command, a localhost would be output by the command. 
Click on the link and refresh as the model trains to visualise the different metrics for training.

```bash
$ tensorboard --logdir=ppo_coupled_approach_reward  # Visualing Paper Implementation Results on PPO Algorithm
$ tensorboard --logdir=ddpg_coupled_approach_reward  # Visualing Paper Implementation Results on DDPG Algorithm

$ tensorboard --logdir=ppo_decoupled_approach_reward  # Visualing Novel Approach Results on PPO Algorithm
```

### Contact:
1. Rahul Rustagi (rustagirahul24@gmail.com)
2. Chinmay Joshi
