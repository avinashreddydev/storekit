# %%

import gymnasium as gym
import envs  # triggers the register() calls in envs/__init__.py

from envs.env_v0 import StoreConfig, Demand

from gymnasium.vector import SyncVectorEnv  # or AsyncVectorEnv
from datetime import date

# --- assume StoreConfig, Demand, StoreEnv are already defined as in your code ---

# 1) Per-env configs & demands (customize as you like)
store_config_1 = StoreConfig(seed=101, start_date=date(2021, 1, 1))
demand_1 = Demand(seed=1001, base=10.0)

store_config_2 = StoreConfig(seed=102, start_date=date(2021, 1, 1))
demand_2 = Demand(seed=1002, base=11.0)

store_config_3 = StoreConfig(seed=103, start_date=date(2021, 1, 1))
demand_3 = Demand(seed=1003, base=9.0)

store_config_4 = StoreConfig(seed=104, start_date=date(2021, 1, 1))
demand_4 = Demand(seed=1004, base=12.0)



env_id = "StoreEnv-v0"


env_fns = [
    lambda : gym.make(env_id, config = store_config_1, demand = demand_1), 
    lambda : gym.make(env_id, config = store_config_2, demand = demand_2), 
    lambda : gym.make(env_id, config = store_config_3, demand = demand_3), 
    lambda : gym.make(env_id, config = store_config_4, demand = demand_4), 
]




def run():
    env = gym.make("StoreEnv-v0")  # kwargs would be passed here if your __init__ accepts any
    obs, info = env.reset()
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print("obs:", obs, "reward:", reward)



def run_vector_envs():
    venv = SyncVectorEnv(env_fns)
    obs, info = venv.reset()




# %%
if __name__ == "__main__":
    pass
    # run()
    # run_vector_envs()
