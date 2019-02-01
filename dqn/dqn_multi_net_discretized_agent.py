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
        # creation of a neural net for every action
        self.eval_action_nets = [self.build_model(self.n_inputs) for _ in range(n_actions)]
        self.target_action_nets = [self.build_model(self.n_inputs) for _ in range(n_actions)]
        self.replace_target_iter = replace_target_iter

        self.replay_counter = 0
        self.win_rates = []
        self.average_rewards = []

    # Creates neural net for DQN
    def build_model(self, n_inputs):
        neural_net = Sequential()
        neural_net.add(Dense(24, input_dim=n_inputs, activation='relu'))
        neural_net.add(Dense(24, activation='relu'))
        neural_net.add(Dense(1, activation='linear'))
        neural_net.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return neural_net

    def update(self, prev_obs, prev_act, obs, reward, episode_reward, done):
        self.remember(prev_obs, prev_act, obs, reward, done)
        if len(self.memory) > self.batch_size:
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
                for act_net in self.target_action_nets:
                    action_predictions.append(act_net.predict(obs)[0])
                target = rew + self.gamma * np.max(action_predictions)
            else:
                target = rew
            # fit predicted value of previous action in previous observation to target value of max_action
            x_batch[prev_act].append(prev_obs[0])
            y_batch[prev_act].append(target)
        for a in range(self.n_actions):
            if len(x_batch[a]) != 0:
                self.eval_action_nets[a].fit(np.array(x_batch[a]), np.array(y_batch[a]), batch_size=len(x_batch[a]), verbose=0)

    # Chooses action with epsilon greedy exploration policy
    def choose_action(self, obs):
        action_predictions = []
        for act_net in self.eval_action_nets:
            action_predictions.append(act_net.predict(obs)[0])
        greedy_action = np.argmax(action_predictions)
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
        if done and episode_reward == 200:
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
