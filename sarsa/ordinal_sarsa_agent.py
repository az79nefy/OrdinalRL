import numpy as np
import matplotlib.pyplot as plt
import random


class SarsaAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_min, n_actions, n_ordinals, n_observations, randomize):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions
        self.n_ordinals = n_ordinals

        # Ordinal_Values (3-dimensional array with ordinal_value (array of floats) for each action in each observation)
        self.ordinal_values = np.full((n_observations, n_actions, n_ordinals), 0.0)

        self.win_rates = []
        self.average_rewards = []

    def update(self, prev_obs, prev_act, obs, act, reward, episode_reward, done):
        ordinal = self.reward_to_ordinal(reward, episode_reward, done)
        # update ordinal_values with received ordinal
        self.update_ordinal_values(prev_obs, prev_act, obs, act, ordinal)

    # Updates ordinal_values based on probability of ordinal reward occurrence for each action
    def update_ordinal_values(self, prev_obs, prev_act, obs, act, ordinal):
        # reduce old data weight
        for i in range(self.n_ordinals):
            self.ordinal_values[prev_obs, prev_act, i] *= (1 - self.alpha)
            self.ordinal_values[prev_obs, prev_act, i] += self.alpha * (self.gamma * self.ordinal_values[obs, act, i])

        # add new data point
        self.ordinal_values[prev_obs, prev_act, ordinal] += self.alpha

    # Computes borda_values for one observation given the ordinal_values
    def compute_borda_scores(self, obs):
        # sum up all ordinal values per action for given observation
        ordinal_value_sum_per_action = np.zeros(self.n_actions)
        for action_a in range(self.n_actions):
            for ordinal_value in self.ordinal_values[obs, action_a]:
                ordinal_value_sum_per_action[action_a] += ordinal_value

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
                        for ordinal_count in range(self.n_ordinals):
                            ordinal_probability_a = self.ordinal_values[obs, action_a, ordinal_count] \
                                                    / ordinal_value_sum_per_action[action_a]
                            # ordinal_probability_b is also the tie probability
                            ordinal_probability_b = (self.ordinal_values[obs, action_b, ordinal_count] /
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
    def choose_action(self, obs, greedy):
        greedy_action = self.get_greedy_action(obs)
        # choose random action with probability epsilon
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        # greedy action is chosen with probability (1 - epsilon)
        else:
            return greedy_action

    def end_episode(self, n_episodes):
        # gradually reduce epsilon after every done episode
        self.epsilon = self.epsilon - 2 / n_episodes if self.epsilon > self.epsilon_min else self.epsilon_min

    def preprocess_observation(self, obs):
        return obs

    # Mapping of reward value to ordinal reward (has to be configured per game)
    def reward_to_ordinal(self, reward, episode_reward, done):
        if reward == -10:
            return 0
        if reward == -1:
            return 1
        else:
            return 2

    # Returns Boolean, whether the win-condition of the environment has been met
    @staticmethod
    def check_win_condition(reward, episode_reward, done):
        if done and reward == 20:
            return True
        else:
            return False

    def evaluate(self, i_episode, episode_rewards, episode_wins):
        # compute average episode reward and win rate over last episodes
        average_reward = round(sum(episode_rewards) / len(episode_rewards), 2)
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
