import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools


'''  CONFIGURATION  '''

env = gym.make('CartPole-v0')

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

# Discretize the observation space (specify manually)
cart_pos_space = np.linspace(-2.4, 2.4, 10)
cart_vel_space = np.linspace(-4, 4, 10)
pole_theta_space = np.linspace(-0.20943951, 0.20943951, 10)
pole_theta_vel_space = np.linspace(-4, 4, 10)
observation_space = [cart_pos_space, cart_vel_space, pole_theta_space, pole_theta_vel_space]


''' INITIALIZATION '''

# Number of possible actions
n_actions = env.action_space.n

# List of all possible discrete observations
observation_range = [range(len(i)+1) for i in observation_space]

# Q_Values (dictionary with float-value for each action and each possible observation)
q_values = {}
for observation in list(itertools.product(*observation_range)):
    for a in range(n_actions):
        if randomize:
            q_values[observation, a] = random.random() / 10
        else:
            q_values[observation, a] = 0


'''  FUNCTION DEFINITION  '''


def get_discrete_observation(obs):
    discrete_observation = []
    for obs_idx in range(len(obs)):
        discrete_observation.append(int(np.digitize(obs[obs_idx], observation_space[obs_idx])))
    return tuple(discrete_observation)


# Returns Boolean, whether the win-condition of the environment has been met
def check_win_condition():
    if done and episode_reward == 200:
        return True
    else:
        return False


# Updates Q_Values based on probability of ordinal reward occurrence for each action
def update_q_values(prev_obs, prev_act, obs, rew):
    possible_q_values = np.array([q_values[obs, act] for act in range(n_actions)])
    q_old = q_values[prev_obs, prev_act]
    q_new = (1 - alpha) * q_old + alpha * (rew + gamma * np.max(possible_q_values))
    q_values[prev_obs, prev_act] = q_new


# Chooses action with epsilon greedy exploration policy
def choose_action(obs):
    possible_q_values = np.array([q_values[obs, act] for act in range(n_actions)])
    greedy_action = np.argmax(possible_q_values)
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
    observation = get_discrete_observation(env.reset())
    action = choose_action(observation)

    prev_observation = None
    prev_action = None

    episode_reward = 0
    for t in range(max_timesteps):
        observation_cont, reward, done, info = env.step(action)
        observation = get_discrete_observation(observation_cont)
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
