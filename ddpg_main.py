import gym
import torch
import simple_driving
import time
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

def main():

    env = gym.make('HardDriving-v0')
    action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))
    model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise ,tensorboard_log=f"./{args.logname}/")
    
    TIME_STEPS = 100_000
    model.learn(total_timesteps=TIME_STEPS,reset_num_timesteps=False, tb_log_name=f"ddpg_{TIME_STEPS}")
    model.save(f"ddpg_{args.model}")
    del model  # delete trained model to demonstrate loading

    # Load the trained agent
    model = DDPG.load(f"ddpg_{args.model}", env=env, print_system_info=True)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        # print(f"Step: {i} Action: {action} Reward: {rewards} Done: {dones}")
        vec_env.render("human")

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--logname", default="ddpg_test")
    parser.add_argument("--model", default="test1")                
    args = parser.parse_args()
    main()
