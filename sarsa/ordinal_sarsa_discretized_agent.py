import numpy as np
import matplotlib.pyplot as plt
import random
import itertools


class SarsaAgent:
    def __init__(self, alpha, gamma, epsilon, n_actions, n_ordinals, n_observations, randomize):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.n_ordinals = n_ordinals
        self.observation_space = self.init_observation_space()
        self.observation_to_index = self.build_obs_dict(self.observation_space)

        if randomize:
            self.borda_values = np.full((n_observations, n_actions), random.random()/10)
        else:
            self.borda_values = np.full((n_observations, n_actions), 0.0)

        self.ordinal_values = np.full((n_observations, n_actions, n_ordinals), 0.0)

        self.win_rates = []
        self.average_rewards = []

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

    def update(self, prev_obs, prev_act, obs, act, reward, episode_reward, done):
        ordinal = self.reward_to_ordinal(reward, episode_reward, done)
        self.update_ordinal_values(prev_obs, prev_act, obs, act, ordinal)
        self.update_borda_scores(prev_obs)

    # Updates ordinal_values based on probability of ordinal reward occurrence for each action
    def update_ordinal_values(self, prev_obs, prev_act, obs, act, ordinal):
        # reduce old data weight
        for i in range(self.n_ordinals):
            self.ordinal_values[prev_obs, prev_act, i] *= (1 - self.alpha)
            self.ordinal_values[prev_obs, prev_act, i] += self.alpha * (self.gamma * self.ordinal_values[obs, act, i])

        # add new data point
        self.ordinal_values[prev_obs, prev_act, ordinal] += self.alpha

    # Updates borda_values for one observation given the ordinal_values
    def update_borda_scores(self, prev_obs):
        # sum up all ordinal values per action for given observation
        ordinal_value_sum_per_action = np.zeros(self.n_actions)
        for action_a in range(self.n_actions):
            for ordinal_value in self.ordinal_values[prev_obs, action_a]:
                ordinal_value_sum_per_action[action_a] += ordinal_value

        # count actions whose ordinal value sum is not zero (no comparision possible for actions without ordinal_value)
        non_zero_action_count = np.count_nonzero(ordinal_value_sum_per_action)
        actions_to_compare_count = non_zero_action_count - 1

        # compute borda_values for action_a (probability that action_a wins against any other action)
        for action_a in range(self.n_actions):
            # if action has not yet recorded any ordinal values, action has to be played (set borda_value to 1.0)
            if ordinal_value_sum_per_action[action_a] == 0:
                self.borda_values[prev_obs, action_a] = 1.0
                continue

            if actions_to_compare_count < 1:
                # set lower than 1.0 (borda_value for zero_actions is 1.0)
                self.borda_values[prev_obs, action_a] = 0.5
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
                        for ordinal_count in range(self.n_ordinals):
                            ordinal_probability_a = self.ordinal_values[prev_obs, action_a, ordinal_count] \
                                                    / ordinal_value_sum_per_action[action_a]
                            # ordinal_probability_b is also the tie probability
                            ordinal_probability_b = (self.ordinal_values[prev_obs, action_b, ordinal_count] /
                                                     ordinal_value_sum_per_action[action_b])
                            winning_probability_a += ordinal_probability_a * \
                                (worse_probability_b + ordinal_probability_b / 2.0)
                            worse_probability_b += ordinal_probability_b
                        winning_probability_a_sum += winning_probability_a
                # normalize summed up probabilities with number of actions that have been compared
                self.borda_values[prev_obs, action_a] = winning_probability_a_sum / actions_to_compare_count

    # Chooses action with epsilon greedy exploration policy
    def choose_action(self, obs):
        greedy_action = np.argmax(self.borda_values[obs])
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
        return self.observation_to_index[tuple(discrete_observation)]

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
        # compute average episode reward and win rate over last 100 episodes
        average_reward = sum(episode_rewards) / len(episode_rewards)
        win_rate = sum(episode_wins) / len(episode_wins)
        # store average episode reward and win rate over last 100 episodes for plotting purposes
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
