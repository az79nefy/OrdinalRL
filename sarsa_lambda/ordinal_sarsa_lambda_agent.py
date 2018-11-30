import numpy as np
import matplotlib.pyplot as plt
import random


class SarsaLambdaAgent:
    def __init__(self, alpha, gamma, epsilon, lambda_, n_actions, n_ordinals, n_observations, randomize):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.n_actions = n_actions
        self.n_ordinals = n_ordinals

        # Borda_Values (2-dimensional array with float-value for each action (e.g. [Left, Down, Right, Up]) in each observation)
        if randomize:
            self.borda_values = np.full((n_observations, n_actions), random.random()/10)
        else:
            self.borda_values = np.full((n_observations, n_actions), 0.0)

        # Ordinal_Values (3-dimensional array with ordinal_value (array of floats) for each action in each observation)
        self.ordinal_values = np.full((n_observations, n_actions, n_ordinals), 0.0)

        # Eligibility Trace (2-dimensional array with float-values for each action in each observation)
        self.eligibility_trace = np.full((n_observations, n_actions), 0.0)
        self.win_rates = []
        self.average_rewards = []

    def update(self, prev_obs, prev_act, obs, act, reward, episode_reward, done):
        # increase eligibility trace entry for executed observation-action pair
        self.eligibility_trace[prev_obs, prev_act] += 1
        ordinal = self.reward_to_ordinal(reward, episode_reward, done)
        # update ordinal_values with received ordinal
        self.update_ordinal_values(prev_obs, prev_act, obs, act, ordinal)
        # update borda_values with updated ordinal_values
        self.update_borda_scores(prev_obs)
        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_

    # Updates ordinal_values based on probability of ordinal reward occurrence for each action
    def update_ordinal_values(self, prev_obs, prev_act, obs, act, ordinal):
        # reduce old data weight
        for i in range(self.n_ordinals):
            rew = 1 if i == ordinal else 0
            ordinal_value_old = self.ordinal_values[prev_obs, prev_act, i]
            ordinal_value_target = rew + self.gamma * self.ordinal_values[obs, act, i]
            self.ordinal_values[:, :, i] += self.alpha * (ordinal_value_target - ordinal_value_old) * self.eligibility_trace

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
        # reset eligibility trace after every episode
        self.eligibility_trace *= 0

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
