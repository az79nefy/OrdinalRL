import gym
import numpy as np
import random

'''  CONFIGURATION  '''

env = gym.make('Taxi-v2')
# learning rate
alpha = 0.5
# discount factor
gamma = 0.9
# epsilon-greedy exploration for action choice
epsilon = 0.05
# flag whether to randomize action estimates at initialization
randomize = True
# maximal timesteps to be used per episode
max_timesteps = 1000
# number of episodes to be run
nEpisodes = 20000

# number of possible actions
n_actions = env.action_space.n
# number of possible states
n_states = env.observation_space.n
# number of ordinals (has to be specified manually)
n_ordinals = 3
# flag whether to use classic version (based on probability of ordinal occurrence)
ordinal_based = True
# position of neutral reward (indexed from zero)
neutral_reward_position = 0


# mapping of reward value to ordinal reward (has to be configured per game)
def reward_to_ordinal(reward_value):
    if reward_value == -10:
        return 0
    if reward_value == -1:
        return 1
    else:
        return 2


''' INITIALIZATION '''

# 2-dimensional array: borda-values (float) for each action (e.g. [Left, Down, Right, Up]) in each state
borda_values = [[1.0 for x in range(n_actions)] for y in range(n_states)]
# 3-dimensional array: ordinal-values (array) for each action in each state
ordinal_values = [[[0.0 for x in range(n_ordinals)] for y in range(n_actions)] for z in range(n_states)]

if not ordinal_based:
    for ordinal_state_values in ordinal_values:
        for ordinal_action_values in ordinal_state_values:
            ordinal_action_values[neutral_reward_position] = 1.0


if randomize:
    borda_values = [[random.random() / 10 for x in range(n_actions)] for y in range(n_states)]


# action choice with epsilon greedy exploration policy
def choose_action(state):
    greedy_action = np.argmax(borda_values[state])
    # non-greedy action is chose with probability epsilon
    if random.random() < epsilon:
        non_greedy_actions = list(range(n_actions))
        non_greedy_actions.remove(greedy_action)
        return random.choice(non_greedy_actions)
    # greedy action is chosen with probability (1 - epsilon)
    else:
        return greedy_action


# update borda_values for one state (given new ordinal_values)
def update_borda_scores(state_borda_values, state_ordinal_values):
    # sum up all ordinal values per action for given state
    ordinal_value_sum_per_action = np.zeros(n_actions)
    for action_a in range(n_actions):
        for ordinal_value in state_ordinal_values[action_a]:
            ordinal_value_sum_per_action[action_a] += ordinal_value

    # count actions whose ordinal value sum is not zero (no comparision possible for actions without ordinal_value)
    non_zero_action_count = np.count_nonzero(ordinal_value_sum_per_action)

    for action_a in range(n_actions):
        # if action has not yet recorded any ordinal values, action has to be played (set borda_value to 1.0)
        if ordinal_value_sum_per_action[action_a] == 0:
            state_borda_values[action_a] = 1.0
            continue

        # compute borda_values for action_a
        # compare action_a to all other actions (compute probability that action wins against any other action)

        # BASIC COMPUTATION
        actions_to_compare_count = non_zero_action_count - 1
        if actions_to_compare_count < 1:
            # set lower than 1.0 (borda_value for zero_actions is 1.0)
            state_borda_values[action_a] = 0.5
        else:
            # over all actions: sum up the probabilities that action_a wins against the given action
            winning_probability_sum = 0
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
                        ordinal_probability_a = state_ordinal_values[action_a][ordinal_count] \
                                                     / ordinal_value_sum_per_action[action_a]
                        # ordinal_probability_b is also the tie probability
                        ordinal_probability_b = (state_ordinal_values[action_b][ordinal_count] /
                                                 ordinal_value_sum_per_action[action_b])
                        winning_probability_a += ordinal_probability_a * (worse_probability_b + ordinal_probability_b / 2.0)
                        worse_probability_b += ordinal_probability_b
                    winning_probability_sum += winning_probability_a
            # normalize summed up probabilities with number of actions that have been compared
            state_borda_values[action_a] = winning_probability_sum / actions_to_compare_count


''' EXECUTION '''

episode_rewards = []
for i_episode in range(nEpisodes):
    observation = env.reset()
    action = choose_action(observation)

    prev_observation = None
    prev_action = None

    for t in range(max_timesteps):
        observation, reward, done, info = env.step(action)
        # next action to be executed (based on new observation)
        action = choose_action(observation)

        if prev_observation is not None:

            # ORDINAL_SCORE-BASED: based on probability of ordinal reward sum per episode for each action
            if not ordinal_based:
                ordinal_new = reward_to_ordinal(reward)
                # shift ordinal to have value 0 at neutral reward
                ordinal_new -= neutral_reward_position

                for i in range(n_ordinals):
                    if done:
                        ordinal_values[prev_observation][prev_action][i] *= (1 - alpha)
                    else:
                        ordinal_values[prev_observation][prev_action][i] *= (1 - alpha)

                        back_prop_value = None
                        if i - ordinal_new < 0 or i - ordinal_new >= n_ordinals:
                            back_prop_value = 0
                        else:
                            # TODO: Extend array into the direction of back-propagation
                            back_prop_value = ordinal_values[observation][action][i - ordinal_new]

                        # back-propagate future ordinal values to previous state and action
                        ordinal_values[prev_observation][prev_action][i] += \
                            alpha * (gamma * ordinal_values[observation][action][i])

            # ORDINAL-BASED: based on probability of ordinal reward occurrence for each action
            else:
                # reduce old data weight
                for i in range(n_ordinals):
                    if done:
                        ordinal_values[prev_observation][prev_action][i] *= (1 - alpha)
                    else:
                        ordinal_values[prev_observation][prev_action][i] *= (1 - alpha)
                        ordinal_values[prev_observation][prev_action][i] += \
                            alpha * (gamma * ordinal_values[observation][action][i])

                # add new data point
                ordinal_new = reward_to_ordinal(reward)
                ordinal_values[prev_observation][prev_action][ordinal_new] += alpha

            # update borda scores
            update_borda_scores(borda_values[prev_observation], ordinal_values[prev_observation])

        prev_observation = observation
        prev_action = action

        if done:
            episode_rewards.append(reward)
            if i_episode % 100 == 99:
                print("Episode {} finished. Average reward since last check: {}".format(i_episode + 1, np.mean(episode_rewards)))
                episode_rewards = []
            break

env.close()
