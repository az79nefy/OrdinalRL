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

        # Q_Values (2-dimensional array with float-value for each action (e.g. [Left, Down, Right, Up]) in each observation)
        if randomize:
            self.q_values = np.full((n_observations, n_actions), random.random()/10)
        else:
            self.q_values = np.full((n_observations, n_actions), 0.0)

        # Eligibility Trace (2-dimensional array with float-values for each action in each observation)
        self.eligibility_trace = np.full((n_observations, n_actions), 0.0)
        self.win_rates = []
        self.average_rewards = []

    def update(self, prev_obs, prev_act, obs, act, reward, episode_reward, done):
        # increase eligibility trace entry for executed observation-action pair
        self.eligibility_trace[prev_obs, :] *= 0
        self.eligibility_trace[prev_obs, prev_act] = 1
        self.update_q_values(prev_obs, prev_act, obs, act, reward)
        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lambda_

    # Updates Q_Values based on received reward
    def update_q_values(self, prev_obs, prev_act, obs, act, rew):
        q_old = self.q_values[prev_obs, prev_act]
        q_target = rew + self.gamma * self.q_values[obs, act]
        self.q_values = self.q_values + self.alpha * (q_target - q_old) * self.eligibility_trace

    # Chooses action with epsilon greedy exploration policy
    def choose_action(self, obs):
        greedy_action = np.argmax(self.q_values[obs])
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
