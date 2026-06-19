import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="human")
obs, info = env.reset()

for i in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if reward != 0:
        print(f"Recompensa: {reward}")
    if terminated or truncated:
        obs, info = env.reset()

env.close()