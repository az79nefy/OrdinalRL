import gym
import numpy as np
import random


'''  CONFIGURATION  '''

env = gym.make('Taxi-v2')

# Learning rate
alpha = 0.1
# Discount factor
gamma = 0.9
# Epsilon in epsilon-greedy exploration (for action choice)
epsilon = 1.0

# Number of episodes to be run
n_episodes = 50000
# Maximal timesteps to be used per episode
max_timesteps = 1000

# Flag whether to randomize action estimates at initialization
randomize = False


''' INITIALIZATION '''

# number of possible actions
n_actions = env.action_space.n
# number of possible observations
n_observations = env.observation_space.n

# Q_Values (2-dimensional array with float-value for each action (e.g. [Left, Down, Right, Up]) in each observation)
if randomize:
    q_values = [[random.random() / 10 for x in range(n_actions)] for y in range(n_observations)]
else:
    q_values = [[1.0 for x in range(n_actions)] for y in range(n_observations)]


'''  FUNCTION DEFINITION  '''


# Updates Q_Values based on probability of ordinal reward occurrence for each action
def update_q_values(prev_obs, prev_act, obs, act, rew):
    q_old = q_values[prev_obs][prev_act]
    q_new = (1 - alpha) * q_old + alpha * (rew + gamma * q_values[obs][act])
    q_values[prev_obs][prev_act] = q_new


# Chooses action with epsilon greedy exploration policy
def choose_action(obs):
    greedy_action = np.argmax(q_values[obs])
    # choose random action with probability epsilon
    if random.random() < epsilon:
        return random.randrange(n_actions)
    # greedy action is chosen with probability (1 - epsilon)
    else:
        return greedy_action


''' EXECUTION '''

episode_rewards = []
for i_episode in range(n_episodes):
    observation = env.reset()
    action = choose_action(observation)

    prev_observation = None
    prev_action = None

    episode_reward = 0
    for t in range(max_timesteps):
        observation, reward, done, info = env.step(action)
        # next action to be executed (based on new observation)
        action = choose_action(observation)
        episode_reward += reward

        if prev_observation is not None:
            update_q_values(prev_observation, prev_action, observation, action, reward)

        prev_observation = observation
        prev_action = action

        if done:
            epsilon -= 2 / n_episodes if epsilon > 0 else 0
            episode_rewards.append(episode_reward)
            if i_episode % 100 == 99:
                print("Episode {} finished. Average reward since last check: {}".format(i_episode + 1, np.mean(episode_rewards)))
                episode_rewards = []
            break

env.close()
