import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import RMSprop
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
        # creation of a neural net for every action
        self.eval_action_nets = [self.build_model(self.n_inputs) for _ in range(n_actions)]
        self.target_action_nets = [self.build_model(self.n_inputs) for _ in range(n_actions)]
        self.replace_target_iter = replace_target_iter

        self.update_counter = 0
        self.replay_counter = 0
        self.win_rates = []
        self.average_rewards = []

    # Creates neural net for DQN
    def build_model(self, n_inputs):
        neural_net = Sequential()
        neural_net.add(Conv2D(32, 8, strides=(4, 4),
                              padding='valid', activation='relu', input_shape=n_inputs, data_format='channels_first'))
        neural_net.add(Conv2D(64, 4, strides=(2, 2),
                              padding='valid', activation='relu', input_shape=n_inputs, data_format='channels_first'))
        neural_net.add(Conv2D(64, 3, strides=(1, 1),
                              padding='valid', activation='relu', input_shape=n_inputs, data_format='channels_first'))
        neural_net.add(Flatten())
        neural_net.add(Dense(512, activation='relu'))
        neural_net.add(Dense(1))
        neural_net.compile(loss='mse', optimizer=RMSprop(lr=self.alpha, rho=0.95, epsilon=0.01), metrics=['accuracy'])
        return neural_net

    def update(self, prev_obs, prev_act, obs, reward, episode_reward, done):
        self.update_counter += 1
        self.remember(prev_obs, prev_act, obs, reward, done)
        if self.update_counter % 10 == 0 and len(self.memory) > self.batch_size:
            self.replay()

    def remember(self, prev_obs, prev_act, obs, rew, d):
        self.memory.append((prev_obs, prev_act, obs, rew, d))

    def replay(self):
        # copy evaluation action nets to target action nets at first replay and then every 200 replay steps
        if self.replay_counter % self.replace_target_iter == 0:
            for a in range(self.n_actions):
                self.target_action_nets[a].set_weights(self.eval_action_nets[a].get_weights())
        self.replay_counter += 1

        mini_batch = random.sample(self.memory, self.batch_size)
        x_batch, y_batch = [[] for _ in range(self.n_actions)], [[] for _ in range(self.n_actions)]
        for prev_obs, prev_act, obs, rew, d in mini_batch:
            if not d:
                action_predictions = []
                for act_net in self.eval_action_nets:
                    action_predictions.append(act_net.predict(self.convert(obs))[0])
                best_act = np.argmax(action_predictions)
                target = rew + self.gamma * self.target_action_nets[best_act].predict(self.convert(obs))[0]
            else:
                target = rew
            # fit predicted value of previous action in previous observation to target value of max_action
            x_batch[prev_act].append(self.convert(prev_obs)[0])
            y_batch[prev_act].append(target)
            self.eval_action_nets[prev_act].fit(self.convert(prev_obs), [[target]], verbose=0)
        for a in range(self.n_actions):
            if len(x_batch[a]) != 0:
                self.eval_action_nets[a].fit(np.array(x_batch[a]),  np.array(y_batch[a]), batch_size=len(x_batch[a]), verbose=0)

    def get_greedy_action(self, obs):
        action_predictions = []
        for act_net in self.eval_action_nets:
            action_predictions.append(act_net.predict(self.convert(obs))[0])
        return np.argmax(action_predictions)

    # Chooses action with epsilon greedy exploration policy
    def choose_action(self, obs):
        greedy_action = self.get_greedy_action(obs)
        # choose random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        # greedy action is chosen with probability (1 - epsilon)
        else:
            return greedy_action

    def end_episode(self, n_episodes):
        # gradually reduce epsilon after every done episode
        self.epsilon = self.epsilon - 2 / n_episodes if self.epsilon > self.epsilon_min else self.epsilon_min

    def preprocess_observation(self, obs):
        return obs

    @staticmethod
    def convert(obs):
        return np.expand_dims(np.asarray(obs).astype(np.float64), axis=0)

    # Returns Boolean, whether the win-condition of the environment has been met
    @staticmethod
    def check_win_condition(reward, episode_reward, done):
        if done and episode_reward > 20:
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
        print("{}\t{}".format(average_reward, win_rate))

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
