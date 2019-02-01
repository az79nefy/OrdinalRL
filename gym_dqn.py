import gym
import gym_wrappers
from dqn.ordinal_dqn_discretized_agent import DQNAgent


'''  ####  CONFIGURATION  ####  '''

''' ENVIRONMENT '''

# Choose environment and adjust agent (by import), n_ordinals, n_observations and observation_dim based on environment
env = gym.make('CartPole-v0')
n_ordinals = 2
observation_dim = 4

''' HYPERPARAMETERS '''

# Adjust further hyperparameters (e.g. learning rate and discount factor)
# alpha: Learning rate
# gamma: Discount factor
# epsilon: Epsilon in epsilon-greedy exploration (probability for random action choice)
# randomize: Flag whether to randomize action estimates at initialization
# observation_dim: Dimensionality of observation provided by environment
# batch_size: Number of memories being replayed in one replay step
# memory_len: Maximum numbers of memories that are stored
# n_actions: Number of possible actions
# n_observations: Number of possible observations
# n_ordinals: Number of ordinals (possible different rewards)
agent = DQNAgent(alpha=0.001, gamma=0.9, epsilon=1.0, epsilon_min=0.1, observation_dim=observation_dim, batch_size=64,
                 memory_len=20000, replace_target_iter=300, n_actions=env.action_space.n, n_ordinals=n_ordinals)

# Number of episodes to be run
n_episodes = 200
# Step size for evaluation
step_size = 10


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
        agent.update(prev_observation, prev_action, observation, reward, episode_reward, done)

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
