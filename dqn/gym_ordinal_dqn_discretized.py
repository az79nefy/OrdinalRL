import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
import itertools


'''  CONFIGURATION  '''

env = gym.make('CartPole-v0')

# Learning rate
alpha = 0.001
# Discount factor
gamma = 0.9
# Epsilon in epsilon-greedy exploration (for action choice)
epsilon = 1.0

# Number of episodes to be run
n_episodes = 200
# Maximal timesteps to be used per episode
max_timesteps = 1000

# Flag whether to randomize action estimates at initialization
randomize = False
# Number of ordinals (possible different rewards)
n_ordinals = 2

# Discretize the observation space (specify manually)
n_observations = 11**4
cart_pos_space = np.linspace(-2.4, 2.4, 10)
cart_vel_space = np.linspace(-4, 4, 10)
pole_theta_space = np.linspace(-0.20943951, 0.20943951, 10)
pole_theta_vel_space = np.linspace(-4, 4, 10)
observation_space = [cart_pos_space, cart_vel_space, pole_theta_space, pole_theta_vel_space]


''' INITIALIZATION '''

# Number of possible actions
n_actions = env.action_space.n

# DQN Parameters
n_inputs = env.observation_space.shape[0]
batch_size = 64
memory = deque(maxlen=20000)

# List of all possible discrete observations
observation_range = [range(len(i)+1) for i in observation_space]

# Dictionary that maps discretized observations to array indices
observation_to_index = {}
index_counter = 0
for observation in list(itertools.product(*observation_range)):
    observation_to_index[observation] = index_counter
    index_counter += 1

# Borda_Values (2-dimensional array with float-value for each action (e.g. [Left, Down, Right, Up]) in each observation)
if randomize:
    borda_values = np.array([[random.random() / 10 for x in range(n_actions)] for y in range(n_observations)])
else:
    borda_values = np.array([[1.0 for x in range(n_actions)] for y in range(n_observations)])

'''  FUNCTION DEFINITION  '''


# Neural Net for DQN
def build_model():
    neural_net = Sequential()
    neural_net.add(Dense(24, input_dim=n_inputs, activation='relu'))
    neural_net.add(Dense(24, activation='relu'))
    neural_net.add(Dense(n_ordinals, activation='linear'))
    neural_net.compile(loss='mse', optimizer=Adam(lr=alpha))
    return neural_net


# Mapping of reward value to ordinal reward (has to be configured per game)
def reward_to_ordinal(reward_value):
    if done and not check_win_condition():
        return 0
    else:
        return int(reward_value)


# Returns Boolean, whether the win-condition of the environment has been met
def check_win_condition():
    if done and episode_reward == 200:
        return True
    else:
        return False


'''
Further tweaks for DQN:
- targetNetwork for prediction of target (copied from normal network every X steps)
target = (rew + gamma * np.max(targetNetwork.predict(obs)[0]))
'''


def remember(prev_obs, prev_act, obs, rew, d):
    memory.append((prev_obs, prev_act, obs, rew, d))


def replay(batch_size):
    mini_batch = random.sample(memory, batch_size)
    for prev_obs, prev_act, obs, ordinal, d in mini_batch:
        obs_index = observation_to_index[tuple(obs[0])]
        greedy_action = np.argmax(borda_values[obs_index])
        if not d:
            target = gamma * action_nets[greedy_action].predict(obs)[0]
            target[ordinal] += 1
        else:
            target = np.zeros(n_ordinals)
            target[ordinal] += 1
        # fit predicted value of previous action in previous observation to target value of max_action
        action_nets[prev_act].fit(prev_obs, [[target]], verbose=0)


def get_discrete_observation(obs):
    discrete_observation = []
    for obs_idx in range(len(obs)):
        discrete_observation.append(int(np.digitize(obs[obs_idx], observation_space[obs_idx])))
    return tuple(discrete_observation)


# Updates borda_values for one observation given the ordinal_values
def update_borda_scores():
    prev_obs_index = observation_to_index[tuple(prev_observation[0])]
    # sum up all ordinal values per action for given observation
    ordinal_value_sum_per_action = np.zeros(n_actions)
    for action_a in range(n_actions):
        for ordinal_value in action_nets[action_a].predict(prev_observation)[0]:
            ordinal_value_sum_per_action[action_a] += ordinal_value

    # count actions whose ordinal value sum is not zero (no comparision possible for actions without ordinal_value)
    non_zero_action_count = np.count_nonzero(ordinal_value_sum_per_action)
    actions_to_compare_count = non_zero_action_count - 1

    # compute borda_values for action_a (probability that action_a wins against any other action)
    for action_a in range(n_actions):
        # if action has not yet recorded any ordinal values, action has to be played (set borda_value to 1.0)
        if ordinal_value_sum_per_action[action_a] == 0:
            borda_values[prev_obs_index, action_a] = 1.0
            continue

        if actions_to_compare_count < 1:
            # set lower than 1.0 (borda_value for zero_actions is 1.0)
            borda_values[prev_obs_index, action_a] = 0.5
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
                    # predict ordinal values for action a and b
                    ordinal_values_a = action_nets[action_a].predict(prev_observation)[0]
                    ordinal_values_b = action_nets[action_b].predict(prev_observation)[0]
                    for ordinal_count in range(n_ordinals):
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
            borda_values[prev_obs_index, action_a] = winning_probability_a_sum / actions_to_compare_count


# Chooses action with epsilon greedy exploration policy
def choose_action(obs):
    obs_index = observation_to_index[tuple(obs[0])]
    greedy_action = np.argmax(borda_values[obs_index])
    # choose random action with probability epsilon
    if random.random() < epsilon:
        return random.randrange(n_actions)
    # greedy action is chosen with probability (1 - epsilon)
    else:
        return greedy_action


''' EXECUTION '''

win_rate = 0
win_rate_list = []

episode_reward_list = []
mean_reward_list = []

# creation of a neural net for every action
action_nets = []
for act in range(n_actions):
    action_nets.append(build_model())

for i_episode in range(n_episodes):
    observation = get_discrete_observation(env.reset())
    observation = np.reshape(observation, [1, n_inputs])
    action = choose_action(observation)

    prev_observation = None
    prev_action = None

    episode_reward = 0
    for t in range(max_timesteps):
        observation_cont, reward, done, info = env.step(action)
        observation = get_discrete_observation(observation_cont)
        observation = np.reshape(observation, [1, n_inputs])
        # next action to be executed (based on new observation)
        action = choose_action(observation)
        episode_reward += reward

        if prev_observation is not None:
            received_ordinal = reward_to_ordinal(reward)
            remember(prev_observation, prev_action, observation, received_ordinal, done)

            if len(memory) > batch_size:
                replay(batch_size)
                # update borda_values with updated ordinal_value predictions
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
            if i_episode % 10 == 9:
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
plt.plot(list(range(10, n_episodes + 10, 10)), win_rate_list)
plt.xlabel('Number of episodes')
plt.ylabel('Win rate')

# plot average score
plt.figure()
plt.plot(list(range(10, n_episodes + 10, 10)), mean_reward_list)
plt.xlabel('Number of episodes')
plt.ylabel('Average score')

plt.show()

env.close()
