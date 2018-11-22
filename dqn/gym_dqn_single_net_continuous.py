import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import random


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


''' INITIALIZATION '''

# Number of possible actions
n_actions = env.action_space.n

# DQN Parameters
n_inputs = env.observation_space.shape[0]
batch_size = 64
memory = deque(maxlen=20000)


'''  FUNCTION DEFINITION  '''


# Neural Net for DQN
def build_model():
    neural_net = Sequential()
    neural_net.add(Dense(24, input_dim=n_inputs, activation='relu'))
    neural_net.add(Dense(24, activation='relu'))
    neural_net.add(Dense(n_actions, activation='linear'))
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
    x_batch, y_batch = [], []
    for prev_obs, prev_act, obs, rew, d in mini_batch:
        prediction = model.predict(prev_obs)
        if not d:
            target = rew + gamma * np.max(model.predict(obs)[0])
        else:
            target = rew
        # fit predicted value of previous action in previous observation to target value of max_action
        prediction[0][prev_act] = target
        x_batch.append(prev_obs[0])
        y_batch.append(prediction[0])
    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)


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

model = build_model()
for i_episode in range(n_episodes):
    observation_cont = env.reset()
    observation = np.reshape(observation_cont, [1, n_inputs])
    action = choose_action(observation)

    prev_observation = None
    prev_action = None

    episode_reward = 0
    for t in range(max_timesteps):
        observation_cont, reward, done, info = env.step(action)
        observation = np.reshape(observation_cont, [1, n_inputs])
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
