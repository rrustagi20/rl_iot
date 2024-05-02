from setuptools import setup

setup(
    name="simple_driving",
    version='0.0.1',
    python_requires='>3.5.2',
    install_requires=['gym=0.21.0',
                      'pybullet=3.2.5',
                      'numpy=1.19.5',
                      'pandas=1.3.3',
                      'matplotlib=3.3.4',
                      'torch=1.10.2',
                      'tensorboard=2.10.0',
                      'stable-baselines3=1.3.0'
                      ]
)
