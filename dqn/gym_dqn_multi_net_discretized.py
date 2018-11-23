import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
import itertools


'''  CONFIGURATION  '''

env = gym.make('CartPole-v0')

# Learning rate
alpha = 0.001
# Discount factor
gamma = 0.9
# Epsilon in epsilon-greedy exploration (for action choice)
epsilon = 1.0

# Number of episodes to be run
n_episodes = 200
# Maximal timesteps to be used per episode
max_timesteps = 1000

# Discretize the observation space (specify manually)
n_observations = 11**4
cart_pos_space = np.linspace(-2.4, 2.4, 10)
cart_vel_space = np.linspace(-4, 4, 10)
pole_theta_space = np.linspace(-0.20943951, 0.20943951, 10)
pole_theta_vel_space = np.linspace(-4, 4, 10)
observation_space = [cart_pos_space, cart_vel_space, pole_theta_space, pole_theta_vel_space]


''' INITIALIZATION '''

# Number of possible actions
n_actions = env.action_space.n

# DQN Parameters
n_inputs = env.observation_space.shape[0]
batch_size = 64
memory = deque(maxlen=20000)

# List of all possible discrete observations
observation_range = [range(len(i)+1) for i in observation_space]

# Dictionary that maps discretized observations to array indices
observation_to_index = {}
index_counter = 0
for observation in list(itertools.product(*observation_range)):
    observation_to_index[observation] = index_counter
    index_counter += 1


'''  FUNCTION DEFINITION  '''


# Neural Net for DQN
def build_model():
    neural_net = Sequential()
    neural_net.add(Dense(24, input_dim=n_inputs, activation='relu'))
    neural_net.add(Dense(24, activation='relu'))
    neural_net.add(Dense(1, activation='linear'))
    neural_net.compile(loss='mse', optimizer=Adam(lr=alpha))
    return neural_net


# Returns Boolean, whether the win-condition of the environment has been met
def check_win_condition():
    if done and episode_reward == 200:
        return True
    else:
        return False


'''
Further tweaks for DQN:
- targetNetwork for prediction of target (copied from normal network every X steps)
target = (rew + gamma * np.max(targetNetwork.predict(obs)[0]))
'''


def remember(prev_obs, prev_act, obs, rew, d):
    memory.append((prev_obs, prev_act, obs, rew, d))


def replay(batch_size):
    mini_batch = random.sample(memory, batch_size)
    for prev_obs, prev_act, obs, rew, d in mini_batch:
        if not d:
            action_predictions = []
            for act_net in action_nets:
                action_predictions.append(act_net.predict(obs)[0])
            target = rew + gamma * np.max(action_predictions)
        else:
            target = rew
        # fit predicted value of previous action in previous observation to target value of max_action
        action_nets[prev_act].fit(prev_obs, [[target]], verbose=0)


def get_discrete_observation(obs):
    discrete_observation = []
    for obs_idx in range(len(obs)):
        discrete_observation.append(int(np.digitize(obs[obs_idx], observation_space[obs_idx])))
    return tuple(discrete_observation)


# Chooses action with epsilon greedy exploration policy
def choose_action(obs):
    action_predictions = []
    for act_net in action_nets:
        action_predictions.append(act_net.predict(obs)[0])
    greedy_action = np.argmax(action_predictions)
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

# creation of a neural net for every action
action_nets = []
for act in range(n_actions):
    action_nets.append(build_model())

for i_episode in range(n_episodes):
    observation = get_discrete_observation(env.reset())
    observation = np.reshape(observation, [1, n_inputs])
    action = choose_action(observation)

    prev_observation = None
    prev_action = None

    episode_reward = 0
    for t in range(max_timesteps):
        observation_cont, reward, done, info = env.step(action)
        observation = get_discrete_observation(observation_cont)
        observation = np.reshape(observation, [1, n_inputs])
        # next action to be executed (based on new observation)
        action = choose_action(observation)
        episode_reward += reward

        if prev_observation is not None:
            remember(prev_observation, prev_action, observation, reward, done)

        if len(memory) > batch_size:
            replay(batch_size)

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
            if i_episode % 10 == 9:
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
plt.plot(list(range(10, n_episodes + 10, 10)), win_rate_list)
plt.xlabel('Number of episodes')
plt.ylabel('Win rate')

# plot average score
plt.figure()
plt.plot(list(range(10, n_episodes + 10, 10)), mean_reward_list)
plt.xlabel('Number of episodes')
plt.ylabel('Average score')

plt.show()

env.close()
