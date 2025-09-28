from gymnasium.envs.registration import register


register(
    id="StoreEnv-v0",
    entry_point="envs.env_v0:StoreEnv",
)

# If you add more files like env_v1.py with class MyEnvV1:
# register(id="MyEnv-v1", entry_point="envs.env_v1:MyEnvV1")