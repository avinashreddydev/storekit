# main.py
import gymnasium as gym
import envs  # triggers the register() calls in envs/__init__.py

def run():
    env = gym.make("StoreEnv-v0")  # kwargs would be passed here if your __init__ accepts any
    obs, info = env.reset()
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print("obs:", obs, "reward:", reward)

if __name__ == "__main__":
    run()