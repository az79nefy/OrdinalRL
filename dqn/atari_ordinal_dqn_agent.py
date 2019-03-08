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
        self.n_ordinals = n_ordinals
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
        neural_net.add(Dense(self.n_ordinals))
        neural_net.compile(loss='mse', optimizer=RMSprop(lr=self.alpha, rho=0.95, epsilon=0.01), metrics=['accuracy'])
        return neural_net

    def update(self, prev_obs, prev_act, obs, reward, episode_reward, done):
        ordinal = self.reward_to_ordinal(reward, episode_reward, done)
        self.update_counter += 1
        self.remember(prev_obs, prev_act, obs, ordinal, done)
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
        for prev_obs, prev_act, obs, ordinal, d in mini_batch:
            greedy_action = self.get_greedy_action(obs)
            if not d:
                target = self.gamma * self.target_action_nets[greedy_action].predict(self.convert(obs))[0]
                target[ordinal] += 1
            else:
                target = np.zeros(self.n_ordinals)
                target[ordinal] += 1
            # fit predicted value of previous action in previous observation to target value of max_action
            x_batch[prev_act].append(self.convert(prev_obs)[0])
            y_batch[prev_act].append(target)
        for a in range(self.n_actions):
            if len(x_batch[a]) != 0:
                self.eval_action_nets[a].fit(np.array(x_batch[a]), np.array(y_batch[a]), batch_size=len(x_batch[a]), verbose=0)

    # Computes borda_values for one observation given the ordinal_values
    def compute_borda_scores(self, obs):
        # sum up all ordinal values per action for given observation
        ordinal_value_sum_per_action = np.zeros(self.n_actions)
        ordinal_values_per_action = [[] for _ in range(self.n_actions)]
        for action_a in range(self.n_actions):
            for ordinal_value in self.eval_action_nets[action_a].predict(self.convert(obs))[0]:
                ordinal_value_sum_per_action[action_a] += ordinal_value
                ordinal_values_per_action[action_a].append(ordinal_value)
        ordinal_values_per_action = np.array(ordinal_values_per_action)

        # count actions whose ordinal value sum is not zero (no comparision possible for actions without ordinal_value)
        non_zero_action_count = np.count_nonzero(ordinal_value_sum_per_action)
        actions_to_compare_count = non_zero_action_count - 1

        borda_scores = []
        # compute borda_values for action_a (probability that action_a wins against any other action)
        for action_a in range(self.n_actions):
            # if action has not yet recorded any ordinal values, action has to be played (set borda_value to 1.0)
            if ordinal_value_sum_per_action[action_a] == 0:
                borda_scores.append(1.0)
                continue

            if actions_to_compare_count < 1:
                # set lower than 1.0 (borda_value for zero_actions is 1.0)
                borda_scores.append(0.5)
            else:
                # over all actions: sum up the probabilities that action_a wins against the given action
                winning_probability_a_sum = 0
                # compare action_a to all other actions
                for action_b in range(self.n_actions):
                    if action_a == action_b:
                        continue
                    # not comparable if action_b has no ordinal_values
                    if ordinal_value_sum_per_action[action_b] == 0:
                        continue
                    else:
                        # probability that action_a wins against action_b
                        winning_probability_a = 0
                        # running ordinal probability that action_b is worse than current investigated ordinal
                        worse_probability_b = 0
                        # predict ordinal values for action a and b
                        ordinal_values_a = ordinal_values_per_action[action_a]
                        ordinal_values_b = ordinal_values_per_action[action_b]
                        for ordinal_count in range(self.n_ordinals):
                            ordinal_probability_a = ordinal_values_a[ordinal_count] \
                                                    / ordinal_value_sum_per_action[action_a]
                            # ordinal_probability_b is also the tie probability
                            ordinal_probability_b = (ordinal_values_b[ordinal_count] /
                                                     ordinal_value_sum_per_action[action_b])
                            winning_probability_a += ordinal_probability_a * \
                                (worse_probability_b + ordinal_probability_b / 2.0)
                            worse_probability_b += ordinal_probability_b
                        winning_probability_a_sum += winning_probability_a
                # normalize summed up probabilities with number of actions that have been compared
                borda_scores.append(winning_probability_a_sum / actions_to_compare_count)
        return borda_scores

    def get_greedy_action(self, obs):
        return np.argmax(self.compute_borda_scores(obs))

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

    # Mapping of reward value to ordinal reward (has to be configured per game)
    def reward_to_ordinal(self, reward, episode_reward, done):
        if reward == -1:
            return 0
        elif reward == 0:
            return 0
        else:
            return 1

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
        print("{}\t{}\t{}".format(i_episode + 1, average_reward, win_rate))

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
