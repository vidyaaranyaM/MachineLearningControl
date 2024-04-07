from reinforce_agent import REINFORCE
import gymnasium as gym


MODEL_PATH = "nn_model.pth"


if __name__=="__main__":
    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]

    agent = REINFORCE(obs_space_dims, action_space_dims)
    agent.load_model(MODEL_PATH)

    obs, info = env.reset()

    max_time_steps = 1000
    done = False
    num_steps = 0
    while not done and num_steps < max_time_steps:
            # action = agent.sample_action(obs)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
