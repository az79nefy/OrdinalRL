import numpy as np
import matplotlib.pyplot as plt
import random


class QAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_min, n_actions, n_ordinals, n_observations, randomize):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

        # Q_Values (2-dimensional array with float-value for each action (e.g. [Left, Down, Right, Up]) in each observation)
        if randomize:
            self.q_values = np.full((n_observations, n_actions), random.random()/10)
        else:
            self.q_values = np.full((n_observations, n_actions), 0.0)

        self.win_rates = []
        self.average_rewards = []

    def update(self, prev_obs, prev_act, obs, reward, episode_reward, done):
        self.update_q_values(prev_obs, prev_act, obs, reward)

    # Updates Q_Values based on received reward
    def update_q_values(self, prev_obs, prev_act, obs, rew):
        q_old = self.q_values[prev_obs, prev_act]
        q_new = (1 - self.alpha) * q_old + self.alpha * (rew + self.gamma * np.max(self.q_values[obs]))
        self.q_values[prev_obs, prev_act] = q_new

    def get_greedy_action(self, obs):
        return np.argmax(self.q_values[obs])

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
