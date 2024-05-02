import gym
import torch
# from agent import TRPOAgent
import simple_driving
import time
from stable_baselines3 import PPO

def main():
    # nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),torch.nn.Linear(64, 2))
    # agent = TRPOAgent(policy=nn)
    # agent.load_model("agent.pth")
    # agent.train("SimpleDriving-v0", seed=0, batch_size=5000, iterations=10, max_episode_length=250, verbose=True)
    # agent.save_model("agent.pth")

    env = gym.make('HardDriving-v0')

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_test/")
    
    TIME_STEPS = 100_000
    # for i in range(10):
    i=1
    model.learn(total_timesteps=TIME_STEPS,reset_num_timesteps=False, tb_log_name=f"ppo_{i*TIME_STEPS}")
    model.save(f"ppo_{i*TIME_STEPS}")
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
