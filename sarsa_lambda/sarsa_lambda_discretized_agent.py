import numpy as np
import matplotlib.pyplot as plt
import random
import itertools


class SarsaLambdaAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_min, lambda_, n_actions, n_ordinals, n_observations, randomize):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.lambda_ = lambda_
        self.n_actions = n_actions
        self.observation_space = self.init_observation_space()
        self.observation_to_index = self.build_obs_dict(self.observation_space)

        # Q_Values (2-dimensional array with float-value for each action (e.g. [Left, Down, Right, Up]) in each observation)
        if randomize:
            self.q_values = np.full((n_observations, n_actions), random.random()/10)
        else:
            self.q_values = np.full((n_observations, n_actions), 0.0)

        # Eligibility Trace (2-dimensional array with float-values for each action in each observation)
        self.eligibility_trace = np.full((n_observations, n_actions), 0.0)
        self.win_rates = []
        self.average_rewards = []

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
        self.epsilon = self.epsilon - 2 / n_episodes if self.epsilon > self.epsilon_min else self.epsilon_min
        # reset eligibility trace after every episode
        self.eligibility_trace *= 0

    def preprocess_observation(self, obs):
        discrete_observation = []
        for obs_idx in range(len(obs)):
            discrete_observation.append(int(np.digitize(obs[obs_idx], self.observation_space[obs_idx])))
        return self.observation_to_index[tuple(discrete_observation)]

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
