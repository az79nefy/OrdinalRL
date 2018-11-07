import gym
import numpy as np
import matplotlib.pyplot as plt
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


# Returns Boolean, whether the win-condition of the environment has been met
def check_win_condition():
    if done and reward == 20:
        return True
    else:
        return False


# Updates Q_Values based on probability of ordinal reward occurrence for each action
def update_q_values(prev_obs, prev_act, obs, rew):
    q_old = q_values[prev_obs][prev_act]
    q_new = (1 - alpha) * q_old + alpha * (rew + gamma * np.max(q_values[obs]))
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

win_rate = 0
win_rate_list = []

episode_reward_list = []
mean_reward_list = []

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
            update_q_values(prev_observation, prev_action, observation, reward)

        prev_observation = observation
        prev_action = action

        if done:
            # gradually reduce epsilon after every done episode
            epsilon -= 2 / n_episodes if epsilon > 0 else 0
            # update reward and win statistics
            episode_reward_list.append(episode_reward)
            if check_win_condition():
                win_rate += 1.0 / 100

            # compute reward and win statistics every 100 episodes
            if i_episode % 100 == 99:
                mean_reward = np.mean(episode_reward_list)
                print("Episode {} finished. Average reward since last check: {}".format(i_episode + 1, mean_reward))
                # store episode reward mean and win rate over last 100 episodes for plotting purposes
                mean_reward_list.append(mean_reward)
                win_rate_list.append(win_rate)
                # reset running reward and win statistics
                episode_reward_list = []
                win_rate = 0
            break

# plot win rate
plt.figure()
plt.plot(list(range(100, n_episodes + 100, 100)), win_rate_list)
plt.xlabel('Number of episodes')
plt.ylabel('Win rate')

# plot average score
plt.figure()
plt.plot(list(range(100, n_episodes + 100, 100)), mean_reward_list)
plt.xlabel('Number of episodes')
plt.ylabel('Average score')

plt.show()

env.close()
