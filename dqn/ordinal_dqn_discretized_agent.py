import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
import itertools


class DQNAgent:
    def __init__(self, alpha, gamma, epsilon, n_actions, n_ordinals, n_observations, observation_dim, batch_size, memory_len, randomize):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.n_ordinals = n_ordinals
        self.n_inputs = observation_dim
        self.observation_space = self.init_observation_space()
        self.observation_to_index = self.build_obs_dict(self.observation_space)

        # Borda_Values (2-dimensional array with float-value for each action (e.g. [Left, Down, Right, Up]) in each observation)
        if randomize:
            self.borda_values = np.full((n_observations, n_actions), random.random()/10)
        else:
            self.borda_values = np.full((n_observations, n_actions), 0.0)

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_len)
        # creation of a neural net for every action
        self.action_nets = [self.build_model(self.n_inputs) for _ in range(n_actions)]

        self.win_rates = []
        self.average_rewards = []

    # Creates neural net for DQN
    def build_model(self, n_inputs):
        neural_net = Sequential()
        neural_net.add(Dense(24, input_dim=n_inputs, activation='relu'))
        neural_net.add(Dense(24, activation='relu'))
        neural_net.add(Dense(self.n_ordinals, activation='linear'))
        neural_net.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return neural_net

    # Defines discrete observation space
    @staticmethod
    def init_observation_space():
        cart_pos_space = np.linspace(-2.4, 2.4, 10)
        cart_vel_space = np.linspace(-4, 4, 10)
        pole_theta_space = np.linspace(-0.20943951, 0.20943951, 10)
        pole_theta_vel_space = np.linspace(-4, 4, 10)
        return [cart_pos_space, cart_vel_space, pole_theta_space, pole_theta_vel_space]

    @staticmethod
    def build_obs_dict(observation_space):
        # List of all possible discrete observations
        observation_range = [range(len(i) + 1) for i in observation_space]
        # Dictionary that maps discretized observations to array indices
        observation_to_index = {}
        index_counter = 0
        for observation in list(itertools.product(*observation_range)):
            observation_to_index[observation] = index_counter
            index_counter += 1
        return observation_to_index

    def update(self, prev_obs, prev_act, obs, reward, episode_reward, done):
        ordinal = self.reward_to_ordinal(reward, episode_reward, done)
        self.remember(prev_obs, prev_act, obs, ordinal, done)
        if len(self.memory) > self.batch_size:
            self.replay()
            self.update_borda_scores(prev_obs)

    def remember(self, prev_obs, prev_act, obs, rew, d):
        self.memory.append((prev_obs, prev_act, obs, rew, d))

    def replay(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        for prev_obs, prev_act, obs, ordinal, d in mini_batch:
            obs_index = self.observation_to_index[tuple(obs[0])]
            greedy_action = np.argmax(self.borda_values[obs_index])
            if not d:
                target = self.gamma * self.action_nets[greedy_action].predict(obs)[0]
                target[ordinal] += 1
            else:
                target = np.zeros(self.n_ordinals)
                target[ordinal] += 1
            # fit predicted value of previous action in previous observation to target value of max_action
            self.action_nets[prev_act].fit(prev_obs, [[target]], verbose=0)

    # Updates borda_values for one observation given the ordinal_values
    def update_borda_scores(self, prev_obs):
        prev_obs_index = self.observation_to_index[tuple(prev_obs[0])]
        # sum up all ordinal values per action for given observation
        ordinal_value_sum_per_action = np.zeros(self.n_actions)
        for action_a in range(self.n_actions):
            for ordinal_value in self.action_nets[action_a].predict(prev_obs)[0]:
                ordinal_value_sum_per_action[action_a] += ordinal_value

        # count actions whose ordinal value sum is not zero (no comparision possible for actions without ordinal_value)
        non_zero_action_count = np.count_nonzero(ordinal_value_sum_per_action)
        actions_to_compare_count = non_zero_action_count - 1

        # compute borda_values for action_a (probability that action_a wins against any other action)
        for action_a in range(self.n_actions):
            # if action has not yet recorded any ordinal values, action has to be played (set borda_value to 1.0)
            if ordinal_value_sum_per_action[action_a] == 0:
                self.borda_values[prev_obs_index, action_a] = 1.0
                continue

            if actions_to_compare_count < 1:
                # set lower than 1.0 (borda_value for zero_actions is 1.0)
                self.borda_values[prev_obs_index, action_a] = 0.5
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
                        ordinal_values_a = self.action_nets[action_a].predict(prev_obs)[0]
                        ordinal_values_b = self.action_nets[action_b].predict(prev_obs)[0]
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
                self.borda_values[prev_obs_index, action_a] = winning_probability_a_sum / actions_to_compare_count

    # Chooses action with epsilon greedy exploration policy
    def choose_action(self, obs):
        obs_index = self.observation_to_index[tuple(obs[0])]
        greedy_action = np.argmax(self.borda_values[obs_index])
        # choose random action with probability epsilon
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        # greedy action is chosen with probability (1 - epsilon)
        else:
            return greedy_action

    def end_episode(self, n_episodes):
        # gradually reduce epsilon after every done episode
        self.epsilon -= 2 / n_episodes if self.epsilon > 0 else 0

    def preprocess_observation(self, obs):
        discrete_observation = []
        for obs_idx in range(len(obs)):
            discrete_observation.append(int(np.digitize(obs[obs_idx], self.observation_space[obs_idx])))
        return np.reshape(discrete_observation, [1, self.n_inputs])

    # Mapping of reward value to ordinal reward (has to be configured per game)
    def reward_to_ordinal(self, reward, episode_reward, done):
        if done and not self.check_win_condition(reward, episode_reward, done):
            return 0
        else:
            return 1

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
