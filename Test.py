import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

for i in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("¡Funciona!")