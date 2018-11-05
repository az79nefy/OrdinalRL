import gym
import numpy as np
import matplotlib.pyplot as plt
import random


'''  CONFIGURATION  '''

env = gym.make('Taxi-v2')

# Learning rate
alpha = 0.1
# Discount factor
gamma = 0.9
# Epsilon in epsilon-greedy exploration (for action choice)
epsilon = 1.0

# Number of episodes to be run
n_episodes = 50000
# Maximal timesteps to be used per episode
max_timesteps = 1000

# Flag whether to randomize action estimates at initialization
randomize = False
# Number of ordinals (possible different rewards)
n_ordinals = 3


'''  FUNCTION DEFINITION  '''


# Mapping of reward value to ordinal reward (has to be configured per game)
def reward_to_ordinal(reward_value):
    if reward_value == -10:
        return 0
    if reward_value == -1:
        return 1
    else:
        return 2


# Returns Boolean, whether the win-condition of the environment has been met
def check_win_condition():
    if done and reward == 20:
        return True
    else:
        return False


# Chooses action with epsilon greedy exploration policy
def choose_action(obs):
    greedy_action = np.argmax(borda_values[obs])
    # choose random action with probability epsilon
    if random.random() < epsilon:
        return random.randrange(n_actions)
    # greedy action is chosen with probability (1 - epsilon)
    else:
        return greedy_action


# Updates ordinal_values based on probability of ordinal reward occurrence for each action
def update_ordinal_values(prev_obs, prev_act, obs, act, ordinal):
    # reduce old data weight
    for i in range(n_ordinals):
        ordinal_values[prev_obs][prev_act][i] *= (1 - alpha)
        ordinal_values[prev_obs][prev_act][i] += alpha * (gamma * ordinal_values[obs][act][i])

    # add new data point
    ordinal_values[prev_obs][prev_act][ordinal] += alpha


# Updates borda_values for one observation given the ordinal_values
def update_borda_scores():
    # sum up all ordinal values per action for given observation
    ordinal_value_sum_per_action = np.zeros(n_actions)
    for action_a in range(n_actions):
        for ordinal_value in ordinal_values[prev_observation][action_a]:
            ordinal_value_sum_per_action[action_a] += ordinal_value

    # count actions whose ordinal value sum is not zero (no comparision possible for actions without ordinal_value)
    non_zero_action_count = np.count_nonzero(ordinal_value_sum_per_action)
    actions_to_compare_count = non_zero_action_count - 1

    # compute borda_values for action_a (probability that action_a wins against any other action)
    for action_a in range(n_actions):
        # if action has not yet recorded any ordinal values, action has to be played (set borda_value to 1.0)
        if ordinal_value_sum_per_action[action_a] == 0:
            borda_values[prev_observation][action_a] = 1.0
            continue

        if actions_to_compare_count < 1:
            # set lower than 1.0 (borda_value for zero_actions is 1.0)
            borda_values[prev_observation][action_a] = 0.5
        else:
            # over all actions: sum up the probabilities that action_a wins against the given action
            winning_probability_a_sum = 0
            # compare action_a to all other actions
            for action_b in range(n_actions):
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
                    for ordinal_count in range(n_ordinals):
                        ordinal_probability_a = ordinal_values[prev_observation][action_a][ordinal_count] \
                                                     / ordinal_value_sum_per_action[action_a]
                        # ordinal_probability_b is also the tie probability
                        ordinal_probability_b = (ordinal_values[prev_observation][action_b][ordinal_count] /
                                                 ordinal_value_sum_per_action[action_b])
                        winning_probability_a += ordinal_probability_a * \
                            (worse_probability_b + ordinal_probability_b / 2.0)
                        worse_probability_b += ordinal_probability_b
                    winning_probability_a_sum += winning_probability_a
            # normalize summed up probabilities with number of actions that have been compared
            borda_values[prev_observation][action_a] = winning_probability_a_sum / actions_to_compare_count


''' INITIALIZATION '''

# number of possible actions
n_actions = env.action_space.n
# number of possible observations
n_observations = env.observation_space.n

# Borda_Values (2-dimensional array with float-value for each action (e.g. [Left, Down, Right, Up]) in each observation)
if randomize:
    borda_values = [[random.random() / 10 for x in range(n_actions)] for y in range(n_observations)]
else:
    borda_values = [[1.0 for x in range(n_actions)] for y in range(n_observations)]

# Ordinal_Values (3-dimensional array with ordinal_value (array of floats) for each action in each observation)
ordinal_values = [[[0.0 for x in range(n_ordinals)] for y in range(n_actions)] for z in range(n_observations)]


''' EXECUTION '''

win_rate = 0
win_rate_list = []

episode_reward_list = []
mean_reward_list = []

for i_episode in range(n_episodes):
    observation = env.reset()
    action = choose_action(observation)

    prev_observation = None
    prev_action = None

    episode_reward = 0
    for t in range(max_timesteps):
        observation, reward, done, info = env.step(action)
        # next action to be executed (based on new observation)
        action = choose_action(observation)
        episode_reward += reward

        if prev_observation is not None:
            received_ordinal = reward_to_ordinal(reward)
            # update ordinal_values with received ordinal
            update_ordinal_values(prev_observation, prev_action, observation, action, received_ordinal)
            # update borda_values with updated ordinal_values
            update_borda_scores()

        prev_observation = observation
        prev_action = action

        if done:
            # gradually reduce epsilon after every done episode
            epsilon -= 2 / n_episodes if epsilon > 0 else 0
            # update reward and win statistics
            episode_reward_list.append(episode_reward)
            if check_win_condition():
                win_rate += 1.0 / 100

            # compute reward and win statistics every 100 episodes
            if i_episode % 100 == 99:
                mean_reward = np.mean(episode_reward_list)
                print("Episode {} finished. Average reward since last check: {}".format(i_episode + 1, mean_reward))
                # store episode reward mean and win rate over last 100 episodes for plotting purposes
                mean_reward_list.append(mean_reward)
                win_rate_list.append(win_rate)
                # reset running reward and win statistics
                episode_reward_list = []
                win_rate = 0
            break

# plot win rate
plt.figure()
plt.plot(list(range(100, n_episodes + 100, 100)), win_rate_list)
plt.xlabel('Number of episodes')
plt.ylabel('Win rate')

# plot average score
plt.figure()
plt.plot(list(range(100, n_episodes + 100, 100)), mean_reward_list)
plt.xlabel('Number of episodes')
plt.ylabel('Average score')

plt.show()

env.close()
