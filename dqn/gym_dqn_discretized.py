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
alpha = 0.1
# Discount factor
gamma = 0.9
# Epsilon in epsilon-greedy exploration (for action choice)
epsilon = 1.0

# Number of episodes to be run
n_episodes = 1000
# Maximal timesteps to be used per episode
max_timesteps = 1000

# Flag whether to randomize action estimates at initialization
randomize = False

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

# List of all possible discrete observations
observation_range = [range(len(i)+1) for i in observation_space]

# Dictionary that maps discretized observations to array indices
observation_to_index = {}
index_counter = 0
for observation in list(itertools.product(*observation_range)):
    observation_to_index[observation] = index_counter
    index_counter += 1


# Neural Net for DQN
def build_model(input_size):
    neural_net = Sequential()
    neural_net.add(Dense(6, input_dim=input_size, activation='relu'))
    neural_net.add(Dense(6, activation='relu'))
    neural_net.add(Dense(n_actions, activation='linear'))
    neural_net.compile(loss='mse', optimizer=Adam(lr=alpha))
    return neural_net


n_inputs = 1
model = build_model(n_inputs)
batch_size = 32
memory = deque(maxlen=2000)


'''  FUNCTION DEFINITION  '''


def get_discrete_observation(obs):
    discrete_observation = []
    for obs_idx in range(len(obs)):
        discrete_observation.append(int(np.digitize(obs[obs_idx], observation_space[obs_idx])))
    return observation_to_index[tuple(discrete_observation)]


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


def remember(prev_obs, prev_act, obs, rew):
    memory.append((prev_obs, prev_act, obs, rew))


def replay(batch_size):
    mini_batch = random.sample(memory, batch_size)
    x_batch = []
    y_batch = []
    for prev_obs, prev_act, obs, rew in mini_batch:
        prediction = model.predict(prev_obs)
        target = (rew + gamma * np.max(model.predict(obs)[0]))
        # fit predicted value of previous action in previous observation to target value of max_action
        prediction[0][prev_act] = target
        x_batch.append(prev_obs[0])
        y_batch.append(prediction[0])
    model.fit(np.array(x_batch), np.array(y_batch), epochs=1, verbose=0)


# Chooses action with epsilon greedy exploration policy
def choose_action(obs):
    greedy_action = np.argmax(model.predict(obs)[0])
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
    observation = get_discrete_observation(env.reset())
    observation = np.reshape(observation, [1, n_inputs])
    action = choose_action(observation)

    prev_observation = None
    prev_action = None

    episode_reward = 0
    for t in range(max_timesteps):
        observation_cont, reward, done, info = env.step(action)
        observation = observation_cont
        observation = get_discrete_observation(observation_cont)
        observation = np.reshape(observation, [1, n_inputs])
        # next action to be executed (based on new observation)
        action = choose_action(observation)
        episode_reward += reward

        if prev_observation is not None:
            remember(prev_observation, prev_action, observation, reward)

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
