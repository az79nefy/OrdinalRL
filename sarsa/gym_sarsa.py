import gym
import numpy as np
import random


'''  CONFIGURATION  '''

env = gym.make('Taxi-v2')

# Learning rate
alpha = 0.5
# Discount factor
gamma = 0.9
# Epsilon in epsilon-greedy exploration (for action choice)
epsilon = 0.05

# Number of episodes to be run
n_episodes = 2000
# Maximal timesteps to be used per episode
max_timesteps = 1000

# Flag whether to randomize action estimates at initialization
randomize = True


''' INITIALIZATION '''

# number of possible actions
n_actions = env.action_space.n
# number of possible states
n_states = env.observation_space.n

# Q_VALUES: (2-dimensional array with float-value for each action (e.g. [Left, Down, Right, Up]) in each state)
q_values = [[1.0 for x in range(n_actions)] for y in range(n_states)]
if randomize:
    q_values = [[random.random() / 10 for x in range(n_actions)] for y in range(n_states)]


'''  FUNCTION DEFINITION  '''


# Q_VALUE UPDATE: based on probability of ordinal reward occurrence for each action
def update_q_values(prev_obs, prev_act, obs, act, rew):
    q_old = q_values[prev_obs][prev_act]
    if done:
        q_new = (1 - alpha) * q_old + alpha * rew
    else:
        q_new = (1 - alpha) * q_old + alpha * (rew + gamma * q_values[obs][act])

    # update q_value
    q_values[prev_obs][prev_act] = q_new


# Chooses action with epsilon greedy exploration policy
def choose_action(state):
    greedy_action = np.argmax(q_values[state])
    # non-greedy action is chose with probability epsilon
    if random.random() < epsilon:
        non_greedy_actions = list(range(0, n_actions))
        non_greedy_actions.remove(greedy_action)
        return random.choice(non_greedy_actions)
    # greedy action is chosen with probability (1 - epsilon)
    else:
        return greedy_action


episode_rewards = []
for i_episode in range(n_episodes):
    observation = env.reset()
    action = choose_action(observation)

    prev_observation = None
    prev_action = None

    for t in range(max_timesteps):
        observation, reward, done, info = env.step(action)
        # next action to be executed (based on new observation)
        action = choose_action(observation)

        if prev_observation is not None:
            update_q_values(prev_observation, prev_action, observation, action, reward)

        prev_observation = observation
        prev_action = action

        if done:
            episode_rewards.append(reward)
            if i_episode % 100 == 99:
                print("Episode {} finished. Average reward since last check: {}".format(i_episode + 1, np.mean(episode_rewards)))
                episode_rewards = []
            break

env.close()
