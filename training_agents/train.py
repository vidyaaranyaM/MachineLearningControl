import gymnasium as gym
import torch
import random
import numpy as np
from reinforce_agent import REINFORCE


MODEL_PATH = "nn_model.pth"


if __name__=="__main__":
    env = gym.make("MountainCarContinuous-v0")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    total_num_episodes = int(5e3)  # Total number of episodes
    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.observation_space.shape[0]
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space.shape[0]
    rewards_over_seeds = []
    seed = 5

    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)
    
    agent.save_model(MODEL_PATH)
