# OrdinalRL
This project provides an experimental evaluation of ordinal reinforcement learning for environments from the OpenAI Gym.

## Installation

Clone this repository:

    git clone https://github.com/az79nefy/OrdinalRL.git

Install the following python packages with `pip install`:

  ```
  gym
  keras
  numpy
  scipy
  tensorflow
  ```
  
## Usage

In order to start an experiment, execute one of these four python scripts:

    python gym_q_learning.py
    python gym_sarsa.py
    python gym_sarsa_lambda.py
    python gym_dqn.py

The used algorithm (Q-learning, Sarsa, Sarsa-Lambda, DQN) can be easily seen from the name of the script.
In order to configure the individual scripts, change the parameters which are described and explained in each individual script.
Additionally add the string "ordinal_" in front of the imported agent package at the top of the script, if you want to use the agent which uses ordinal rewards.

## Environments

OpenAI Gym provides a number of different environments. 
In order to adjust the agents to the environments, the following changes have to be done in the main scripts:

- Change the definition of the environment in the 'ENVIRONMENT'-section of the script
- Change the string "discretized_agent" to "agent" of the imported agent package if the environment is already discretized

Afterwards open the python script of the imported agent and adjust following functions:

- If environment is undiscretized: Change the `init_observation_space()` function to the respective code fragment in the `discretized_configs` file. For a new environment define this function accordingly
- If agent is ordinal: Change the `reward_to_ordinal()` function according to the rewards of the environment.
- If agent is not ordinal: Change the `remap_reward()` function if desired
- Change the `check_win_condition()` function according to the win condition of the environment
