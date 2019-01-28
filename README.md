## source
This repository is originally forked from [openai_example_projects](https://bitbucket.org/theconstructcore/openai_examples_projects/src/master/) and published to provide a reinforcement learning environment based on ROS and openai gym.

## dependency
openai_examples_projects depends on the following projects, and baselines in /scripts requires python3 (>=3.5) with the development headers. It's recommended to launch the environment visualized in gazebo firstly and work on a virtualenv to run scripts for training.
1. [openAI gym](https://github.com/openai/gym)
2. [RotorS](https://github.com/zhuzd12/rotors_simulator)
3. [mav_control_rw](https://github.com/zhuzd12/mav_control_rw)
4. [openai_ros](https://github.com/zhuzd12/openai_ros)

## tasks
Currently this package only support navigation task for quadrotor [Pelican](https://github.com/ethz-asl/rotors_simulator/tree/master/rotors_description/urdf). The RL algorithms in baselines try to learn a position controller or an end-to-end controller to navigate the quadrotor from a random position to a target position.

## baselines
The baselines is revised from [spinningup](https://github.com/openai/spinningup) and currently support the following RL algorithms.
1. [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
2. [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
3. [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
4. DAGGER
5. MBMF
