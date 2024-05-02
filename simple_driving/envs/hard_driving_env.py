import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pybullet as p
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt


class HardDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, -.6], dtype=np.float32),
            high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10, -10, -1, -1, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([10, 10, 1, 1, 5, 5, 10, 10], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        # Intialise the standard environment parameters
        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.max_episode_length = 250
        self.curr_steps = 0
        self.episode_rewards = 0
        self.episode_rewards_array = []
        self.total_epsiodes=0

        # self.aoc_array = np.load('existing_array.npy')
        # self.epsiode_array = np.load('existing_episodes.npy')
        # Intialise the IoT Environment parameters
        self.N=10
        self.battery = np.random.uniform(low=0, high=100, size=self.N) # Ensures that the battery is always above 90% intially
        self.battery_threshold = 5.0
        self.decay = np.ones(self.N) * 0.5
        self.nodes = (np.random.rand(self.N,2) * 20) - (np.ones((self.N,2)) * 10) # Randomly initialise the nodes position
        self.alpha = 10 # Corresponds to a dominant decaying when 'd' = 4.5
        self.AoC = np.zeros(self.N) # Age of Charging

        self.reset()
    
    def target_pos(self):
        weights = (100 - self.battery) / 100
        target = np.array([(np.dot(weights, self.nodes[:,0])), (np.dot(weights, self.nodes[:,1]))])
        return (target[0], target[1])   # Return the target position in (x,y) format

    def update_battery(self, pos_agent):
        distances = np.array([np.linalg.norm(node - pos_agent) for node in self.nodes])
        battery_change = (self.alpha / distances**2) - self.decay
        for i in range(0,self.N):
            if battery_change[i] < 0:
                self.AoC[i] += 0.01
            else:
                self.AoC[i] = 0
        self.battery += battery_change

    def step(self, action):
        # Feed action to the car and get observation of car's state
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()

        # Battery Update Step
        self.update_battery(pos_agent=np.array([car_ob[0], car_ob[1]]))

        # Target Update Step
        self.goal = self.target_pos()

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        # reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        # reward = max(self.prev_dist_to_goal - dist_to_goal, 0) - np.sum(self.AoC) # Perturbation in a local pit
        reward = - np.sum(self.AoC) # The goal is to maximise the Age of Charging
        self.episode_rewards += reward
        self.prev_dist_to_goal = dist_to_goal

        # Done by running off boundaries
        # if (car_ob[0] >= 10 or car_ob[0] <= -10 or car_ob[1] >= 10 or car_ob[1] <= -10):
        #     print("Car ran off boundaries")
        #     self.done = True
        
        # Done by reaching goal
        if dist_to_goal < 2:
            # self.done = True
            # print("Car reached goal")
            reward = 30
        
        # Done by curr_steps exceeding max_episode_length
        if self.curr_steps >= self.max_episode_length:
            print("curr_steps exceeded max_episode_length")
            self.done = True
        
        # Done by battery level falling below threshold
        if np.any(self.battery < self.battery_threshold):
            self.done = True

        if(self.done):
            self.total_epsiodes += 1
            self.episode_rewards_array.append(self.episode_rewards)

        self.curr_steps +=1
        ob = np.array(car_ob + self.goal, dtype=np.float32)
        return ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self.client)
        self.car = Car(self.client)

        # Reset the steps
        self.episode_rewards = 0
        self.curr_steps = 0
        # self.AoC = np.zeros(self.N) # Age of Charging
        # Reset the battery levels
        self.battery = np.random.uniform(low=0, high=100, size=self.N) # Ensures that the battery is always above 90% intially

        # Randomly initialise the nodes position
        self.nodes = (np.random.rand(self.N,2) * 20) - (np.ones((self.N,2)) * 10)

        # Set the goal to a random target
        # x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
        #      self.np_random.uniform(-5, -9))
        # y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
        #      self.np_random.uniform(-5, -9))
        target_index = np.argmin(self.battery)
        x = self.nodes[target_index][0]
        y = self.nodes[target_index][1]
        self.goal = (x, y)
        self.done = False

        # Randomly initialise the charging array 
        self.charging = np.random.randint(0, 100, size=10)   # [low,high), size=10

        # Visual element of the goal
        # Goal(self.client, self.goal)

        # Get observation to return
        car_ob = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))
        return np.array(car_ob + self.goal, dtype=np.float32)

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Visual element of the goal
        Goal(self.client, self.target_pos())

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(0.001)

    def close(self):
        print("Training Done")
        # print(self.total_epsiodes)
        # print(len(self.episode_rewards_array))
        x=np.arange(0,self.total_epsiodes)
        plt.plot(x,self.episode_rewards_array)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward vs Episode')
        plt.grid()
        plt.show()
        p.disconnect(self.client)
