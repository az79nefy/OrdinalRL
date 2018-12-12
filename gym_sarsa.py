import gym
from sarsa.ordinal_sarsa_agent import SarsaAgent


'''  ####  CONFIGURATION  ####  '''

''' ENVIRONMENT '''

# Choose environment and adjust agent (import at the top), n_ordinals and n_observations based on environment
env = gym.make('Taxi-v2')
n_ordinals = 3
n_observations = env.observation_space.n

''' HYPERPARAMETERS '''

# Adjust further hyperparameters (e.g. learning rate and discount factor)
# alpha: Learning rate
# gamma: Discount factor
# epsilon: Epsilon in epsilon-greedy exploration (probability for random action choice)
# randomize: Flag whether to randomize action estimates at initialization
# n_actions: Number of possible actions
# n_ordinals: Number of ordinals (possible different rewards)
# n_observations: Number of possible observations
agent = SarsaAgent(alpha=0.1, gamma=0.9, epsilon=1.0, randomize=False,
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
    prev_action = agent.choose_action(prev_observation)

    while True:
        observation, reward, done, info = env.step(prev_action)
        observation = agent.preprocess_observation(observation)
        # next action to be executed (based on new observation)
        action = agent.choose_action(observation)
        episode_reward += reward
        agent.update(prev_observation, prev_action, observation, action, reward, episode_reward, done)

        prev_observation = observation
        prev_action = action

        if done:
            agent.end_episode(n_episodes)
            # update reward and win statistics
            episode_rewards.append(episode_reward)
            episode_wins.append(float(agent.check_win_condition(reward, episode_reward, done)))

            if i_episode % step_size == step_size-1:
                agent.evaluate(i_episode, episode_rewards, episode_wins)
                # reset running reward and win statistics
                episode_rewards = []
                episode_wins = []
            break

agent.plot(n_episodes, step_size)

env.close()
