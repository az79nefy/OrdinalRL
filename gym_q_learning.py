import gym
from q_learning.ordinal_q_discretized_agent import QAgent


'''  ####  CONFIGURATION  ####  '''

''' ENVIRONMENT '''

# Choose environment and adjust agent (import at the top), n_ordinals and n_observations based on environment
env = gym.make('CartPole-v0')
n_ordinals = 2
n_observations = 11**4

''' HYPERPARAMETERS '''

# Adjust further hyperparameters (e.g. learning rate and discount factor)
# alpha: Learning rate
# gamma: Discount factor
# epsilon: Epsilon in epsilon-greedy exploration (probability for random action choice)
# randomize: Flag whether to randomize action estimates at initialization
# n_actions: Number of possible actions
# n_ordinals: Number of ordinals (possible different rewards)
# n_observations: Number of possible observations
agent = QAgent(alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.1, randomize=True,
               n_actions=env.action_space.n, n_ordinals=n_ordinals, n_observations=n_observations)

# Number of episodes to be run
n_episodes = 50000
# Step size for evaluation
step_size = 100


'''  ####  EXECUTION  ####  '''

episode_wins = []
episode_rewards = []

for i_episode in range(n_episodes):
    episode_reward = 0
    prev_observation = agent.preprocess_observation(env.reset())
    prev_action = agent.choose_action(prev_observation, greedy=i_episode % step_size == step_size-1)

    while True:
        observation, reward, done, info = env.step(prev_action)
        observation = agent.preprocess_observation(observation)
        # next action to be executed (based on new observation)
        action = agent.choose_action(observation, greedy=i_episode % step_size == step_size-1)
        episode_reward += reward
        agent.update(prev_observation, prev_action, observation, reward, episode_reward, done)

        prev_observation = observation
        prev_action = action

        if done:
            agent.end_episode(n_episodes)
            if i_episode % step_size == step_size-1:
                print("{}\t{}\toptimal".format(i_episode + 1, episode_reward))
                agent.evaluate(i_episode, episode_rewards, episode_wins)
                # reset running reward and win statistics
                episode_rewards = []
                episode_wins = []
            # update reward and win statistics
            else:
                episode_rewards.append(episode_reward)
                episode_wins.append(float(agent.check_win_condition(reward, episode_reward, done)))
            break

agent.plot(n_episodes, step_size)

env.close()
