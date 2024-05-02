from gym.envs.registration import register

register(
    id='SimpleDriving-v0',
    entry_point='simple_driving.envs:SimpleDrivingEnv'
)

register(
    id='HardDriving-v0',
    entry_point='simple_driving.envs:HardDrivingEnv'
)