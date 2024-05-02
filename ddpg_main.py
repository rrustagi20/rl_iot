import gym
import torch
import simple_driving
import time
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def main():

    env = gym.make('HardDriving-v0')
    action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))
    model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise ,tensorboard_log="./ddpg_test/")
    
    TIME_STEPS = 100_000
    # for i in range(10):
    i=1
    model.learn(total_timesteps=TIME_STEPS,reset_num_timesteps=False, tb_log_name=f"ddpg_{i*TIME_STEPS}")
    model.save(f"ddpg_{i*TIME_STEPS}")
    # model.learn(total_timesteps=100_000)

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render()
        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()


    # ob = env.reset()
    # while True:
    #     action = agent(ob)
    #     ob, _, done, _ = env.step(action)
    #     env.render()
    #     if done:
    #         ob = env.reset()
    #         time.sleep(1/30)

if __name__ == '__main__':
    main()
