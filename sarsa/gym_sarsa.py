import gym
import numpy as np
import random

'''  CONFIGURATION  '''

env = gym.make('FrozenLake-v0')
# learning rate
alpha = 0.5
# discount factor
gamma = 0.9
# epsilon-greedy exploration for action choice
epsilon = 0.05
# flag whether to randomize action estimates at initialization
randomize = True
# maximal timesteps to be used per episode
max_timesteps = 1000
# number of episodes to be run
nEpisodes = 2000

# number of possible actions
nActions = env.action_space.n
# number of possible states
nStates = env.observation_space.n

''' EXECUTION '''

# 2-dimensional array: q-values for each action (e.g. [Left, Down, Right, Up]) in each state
q_values = [[1.0 for x in range(nActions)] for y in range(nStates)]
if randomize:
    q_values = [[random.random()/10 for x in range(nActions)] for y in range(nStates)]


# action choice with epsilon greedy exploration policy
def choose_action(state):
    greedy_action = np.argmax(q_values[state])
    # non-greedy action is chose with probability epsilon
    if random.random() < epsilon:
        non_greedy_actions = list(range(0, nActions))
        non_greedy_actions.remove(greedy_action)
        return random.choice(non_greedy_actions)
    # greedy action is chosen with probability (1 - epsilon)
    else:
        return greedy_action


episode_rewards = []
for i_episode in range(nEpisodes):
    observation = env.reset()
    action = choose_action(observation)

    prev_observation = None
    prev_action = None

    for t in range(max_timesteps):
        observation, reward, done, info = env.step(action)
        # next action to be executed (based on new observation)
        action = choose_action(observation)

        if prev_observation is not None:
            q_old = q_values[prev_observation][prev_action]
            if done:
                q_new = (1-alpha) * q_old + alpha * reward
            else:
                q_new = (1-alpha) * q_old + alpha * (reward + gamma * q_values[observation][action])

            # update q_value
            q_values[prev_observation][prev_action] = q_new

        prev_observation = observation
        prev_action = action

        if done:
            episode_rewards.append(reward)
            if i_episode % 100 == 99:
                print("Episode {} finished. Average reward since last check: {}".format(i_episode + 1, np.mean(episode_rewards)))
                episode_rewards = []
            break

env.close()
