#!/usr/bin/env python3

import gym
import tensorflow as tf
import time
from distutils.dir_util import copy_tree
import os
import os.path as osp
import json
import yaml
import policy_core

import random
import numpy as np
import memory
from openai_ros.task_envs.pelican import pelican_attitude_controller
from openai_ros.task_envs.pelican import pelican_randomquat_position_controller
from openai_ros.task_envs.pelican import pelican_attitude_controller_test

# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from planning_msgs.srv import AttitudeControllerService
import pyquaternion
from spinup.utils.run_utils import setup_logger_kwargs

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DAGGER agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

def call_mpc(env, mpc_service):
    odo = env.unwrapped.get_odom()
    target_pose = env.unwrapped.get_target_pose()
    try:
        res = mpc_service(odo, target_pose)
    except (rospy.ServiceException) as e:
        print("/pelican/call_nlmpc_controller service call failed")
    return [res.cmd.yaw_rate, res.cmd.roll, res.cmd.pitch, res.cmd.thrust.z]

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        os.unlink(file)

if __name__ == '__main__':
    rospy.init_node('pelican_ppo_attitude_controller_dagger_training', anonymous=True, log_level=rospy.WARN)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PelicanAttControllerEnv-v0')
    parser.add_argument('--test_env', type=str, default='PelicanAttControllerTestEnv-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--pretrain_epochs', type=int, default=500)
    parser.add_argument('--update_policy_epochs', type=int, default=5)
    parser.add_argument('--test_frequency', type=int, default=50)
    parser.add_argument('--test_epochs', type=int, default=91)
    parser.add_argument('--activation', type=str, default=tf.tanh)
    parser.add_argument('--output_activation', type=str, default=tf.tanh)
    parser.add_argument('--yaml_file', type=str, default="dagger")
    parser.add_argument('--exp_name', type=str, default='Pelican_position_controller_dagger')
    args = parser.parse_args()

    # env = gym.make('PelicanAttControllerEnv-v0')
    env = gym.make(args.env)
    A_DIM = env.unwrapped.a_dim
    S_DIM = env.unwrapped.s_dim
    # if args.test_env is not None:
    #     test_env = gym.make(args.test_env)
    #     assert(A_DIM == test_env.unwrapped.a_dim)
    #     assert(S_DIM == test_env.unwrapped.s_dim)
    # else:
    #     test_env = env
    # test_env = env
    
    rospy.loginfo("Gym environment done")
    outdir = '/tmp/openai_ros_experiments/'
    env = gym.wrappers.Monitor(env, outdir, resume=True)
    # test_env = gym.wrappers.Monitor(test_env, outdir, resume=True)
    call_nlmpc = rospy.ServiceProxy('/pelican/call_nlmpc_controller', AttitudeControllerService)

    yaml_path = os.path.abspath('yaml_files/'+args.yaml_file+'.yaml')
    assert(os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    
    # set train epoch parameters
    

    epochs = args.epochs
    pre_train_epochs = args.pretrain_epochs
    update_policy_epochs = args.update_policy_epochs
    trainning_policy_epochs = args.test_frequency
    test_policy_epochs = args.test_epochs
    episode_steps = 500

    # load polocy network parameters
    I_learningRate = params["learning_rate"]
    minibatch_size = params["batch_size"]
    discountFactor = params["discount_factor"]
    explorationRate = params["exploration_rate"]
    memorySize = params["memory_size"]
    hidden_sizes = params['hidden_sizes']
    act_noise_amount = params["action_noise"]
    policy_net_training_steps = params["training_steps"]
    # print("hidden_sizes: ", hidden_sizes)


    DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir = DEFAULT_DATA_DIR)
    policy_net = policy_core.Policy_Net(env, EP_MAX=epochs, EP_LEN=episode_steps, GAMMA=discountFactor, LR=I_learningRate, BATCH=minibatch_size, UPDATE_STEP=policy_net_training_steps, 
        hidden_sizes=hidden_sizes, activation=args.activation, output_activation=args.output_activation, act_noise_amount=act_noise_amount, logger_kwargs=logger_kwargs)

    replay_buffer = ReplayBuffer(S_DIM, A_DIM, memorySize)

    start_time = time.time()

    # start iterating from 'current epoch'.
    for epoch in range(epochs):
        env.unwrapped._set_test_mode(False)
        observation = env.reset()
        cumulated_reward = 0
        # buffer_s, buffer_a, buffer_r = [], [], []
        # number of timesteps
        for t in range(episode_steps):
            # action = env.action_space.sample()
            # action = ppo.choose_action(np.array(observation))
            ref_action = call_mpc(env, call_nlmpc)
            if(epoch < pre_train_epochs):
                action = ref_action
            else:
                action = policy_net.choose_action(np.array(observation), True)

            newObservation, reward, done, info = env.step(action)

            replay_buffer.store(observation, ref_action, reward, newObservation, done)
            observation = newObservation

            if (t == episode_steps-1):
                print ("reached the end")
                done = True

            env._flush(force=True)
            cumulated_reward += reward
            
            if done:
                break
            
        # update policy
        if epoch % update_policy_epochs == 0 and epoch > 0:
            for _ in range(20):
                batch = replay_buffer.sample_batch(minibatch_size)
                policy_net.update(batch['obs1'], batch['acts'])

        # test policy
        if epoch % trainning_policy_epochs == 0 and epoch > 0:
            env.unwrapped._set_test_mode(True)
            for i in range(test_policy_epochs):
                observation = env.reset()
                policy_cumulated_reward = 0
                p_buffer_s, p_buffer_a = [], []
                for t in range(episode_steps):
                    action = policy_net.choose_action(np.array(observation), False)
                    
                    # odo = env.unwrapped.get_odom()
                    # print(odo.pose.pose.position)
                    # target_pose = env.unwrapped.get_target_pose()
                    # print("target pose: ", target_pose)
                    # action = call_mpc(env, call_nlmpc)
                    # print("observation: ", observation[9:12])
                    # print("action: ", action)
                    newObservation, reward, done, info = env.step(action)
                    # replay_buffer.store(observation, ref_action, reward, newObservation, done)
                    observation = newObservation
                    if (t == episode_steps-1):
                        print ("reached the end")
                        done = True
                    env._flush(force=True)
                    policy_cumulated_reward += reward

                    if done:
                        m, s = divmod(int(time.time() - start_time), 60)
                        h, m = divmod(m, 60)
                        # print("EP " + str(i) + " - {} test number".format(epoch // trainning_policy_epochs) + " - CReward: " + str(round(policy_cumulated_reward, 4)) + "  Time: %d:%02d:%02d" % (h, m, s))
                        policy_net.logger.store(policy_reward=policy_cumulated_reward)
                        policy_net.logger.store(policy_steps=t)
                        break
                    else:
                        policy_net.logger.store(action_error=info['action_error'])
            
            policy_net.logger.log_tabular('epoch', epoch)
            policy_net.logger.log_tabular('policy_reward', average_only=True)
            policy_net.logger.log_tabular('policy_steps', average_only=True)
            policy_net.logger.log_tabular('action_error', with_min_and_max=False)
            policy_net.logger.log_tabular('LossPi', average_only=True)
            policy_net.logger.dump_tabular()
            # save model
            policy_net.logger.save_state({}, None)

    env.close()