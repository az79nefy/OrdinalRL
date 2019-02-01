import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import random


class DQNAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_min, n_actions, n_ordinals, observation_dim, batch_size, memory_len, replace_target_iter):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions
        self.n_inputs = observation_dim

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_len)
        self.eval_model = self.build_model(self.n_inputs)
        self.target_model = self.build_model(self.n_inputs)
        self.replace_target_iter = replace_target_iter

        self.replay_counter = 0
        self.win_rates = []
        self.average_rewards = []

    # Creates neural net for DQN
    def build_model(self, n_inputs):
        neural_net = Sequential()
        neural_net.add(Dense(6, input_dim=n_inputs, activation='relu'))
        neural_net.add(Dense(6, activation='relu'))
        neural_net.add(Dense(self.n_actions, activation='linear'))
        neural_net.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return neural_net

    def update(self, prev_obs, prev_act, obs, reward, episode_reward, done):
        self.remember(prev_obs, prev_act, obs, reward, done)
        if len(self.memory) > self.batch_size:
            self.replay()

    def remember(self, prev_obs, prev_act, obs, rew, d):
        self.memory.append((prev_obs, prev_act, obs, rew, d))

    def replay(self):
        # copy evaluation model to target model at first replay and then every 200 replay steps
        if self.replay_counter % self.replace_target_iter == 0:
            self.target_model.set_weights(self.eval_model.get_weights())
        self.replay_counter += 1

        mini_batch = random.sample(self.memory, self.batch_size)
        x_batch, y_batch = [], []
        for prev_obs, prev_act, obs, rew, d in mini_batch:
            prediction = self.eval_model.predict(prev_obs)
            if not d:
                target = rew + self.gamma * np.max(self.target_model.predict(obs)[0])
            else:
                target = rew
            # fit predicted value of previous action in previous observation to target value of max_action
            prediction[0][prev_act] = target
            x_batch.append(prev_obs[0])
            y_batch.append(prediction[0])
        self.eval_model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    # Chooses action with epsilon greedy exploration policy
    def choose_action(self, obs):
        greedy_action = np.argmax(self.eval_model.predict(obs)[0])
        # choose random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        # greedy action is chosen with probability (1 - epsilon)
        else:
            return greedy_action

    def end_episode(self, n_episodes):
        # gradually reduce epsilon after every done episode
        self.epsilon = self.epsilon - 2 / n_episodes if self.epsilon > self.epsilon_min else self.epsilon_min

    @staticmethod
    def preprocess_observation(obs):
        return np.expand_dims(obs, axis=0)

    # Returns Boolean, whether the win-condition of the environment has been met
    @staticmethod
    def check_win_condition(reward, episode_reward, done):
        if done and reward == 20:
            return True
        else:
            return False

    def evaluate(self, i_episode, episode_rewards, episode_wins):
        # compute average episode reward and win rate over last episodes
        average_reward = sum(episode_rewards) / len(episode_rewards)
        win_rate = sum(episode_wins) / len(episode_wins)
        # store average episode reward and win rate over last episodes for plotting purposes
        self.average_rewards.append(average_reward)
        self.win_rates.append(win_rate)
        print("Episode {} finished. Average reward since last check: {}".format(i_episode + 1, average_reward))

    # Plots win rate and average score over all episodes
    def plot(self, n_episodes, step_size):
        # plot win rate
        plt.figure()
        plt.plot(list(range(step_size, n_episodes + step_size, step_size)), self.win_rates)
        plt.xlabel('Number of episodes')
        plt.ylabel('Win rate')

        # plot average score
        plt.figure()
        plt.plot(list(range(step_size, n_episodes + step_size, step_size)), self.average_rewards)
        plt.xlabel('Number of episodes')
        plt.ylabel('Average score')

        plt.show()
