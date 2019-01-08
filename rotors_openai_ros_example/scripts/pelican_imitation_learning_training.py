#!/usr/bin/env python3

import gym
import time
from distutils.dir_util import copy_tree
import os
import json
import policy_core

import random
import numpy as np
from keras.models import Sequential, load_model
from keras.initializers import normal
from keras import optimizers
from keras.optimizers import RMSprop
from keras.layers import Convolution2D, Flatten, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import SGD , Adam
import memory
from openai_ros.task_envs.pelican import pelican_attitude_controller

# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from planning_msgs.srv import AttitudeControllerService
import pyquaternion
import utils.logx
from utils.run_utils import setup_logger_kwargs

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
    rospy.init_node('pelican_ppo_attitude_controller_imitation_learning_training', anonymous=True, log_level=rospy.WARN)
    env = gym.make('PelicanAttControllerEnv-v0')
    rospy.loginfo("Gym environment done")
    outdir = '/tmp/openai_ros_experiments/'
    call_nlmpc = rospy.ServiceProxy('/pelican/call_nlmpc_controller', AttitudeControllerService)

    continue_execution = False
    # fill this if continue_execution=True
    weights_path = '/tmp/pelican_AttController_il_ep200.h5'
    monitor_path = '/tmp/pelican_AttController_il_ep200'
    params_json  = '/tmp/pelican_AttController_il_ep200.json'

    A_DIM = env.unwrapped.a_dim
    S_DIM = env.unwrapped.s_dim
    epochs = 20000
    episode_steps = 500
    trainning_policy_epochs = 1000
    test_policy_epochs = 20

    if not continue_execution:
        I_learningRate = 1e-4
        minibatch_size = 32
        A_learningRate = 1e-4
        C_learningRate = 2e-4
        discountFactor = 0.9
        explorationRate = 1
        memorySize = 1000000
        learnStart = 64
        current_epoch = 0
        stepCounter = 0
        loadsim_seconds = 0

        env = gym.wrappers.Monitor(env, outdir, force=True)
    else:
        #Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            minibatch_size = d.get('minibatch_size')
            A_learningRate = d.get('A_learningRate')
            C_learningRate = d.get('C_learningRate')
            discountFactor = d.get('discountFactor')
            explorationRate = d.get('explorationRate')
            learnStart = d.get('learnStart')
            memorySize = d.get('memorySize')
            current_epoch = d.get('current_epoch')
            stepCounter = d.get('stepCounter')
            loadsim_seconds = d.get('loadsim_seconds')


        clear_monitor_files(outdir)
        copy_tree(monitor_path, outdir)
        env = gym.wrappers.Monitor(env, outdir, resume=True)
    logger_kwargs = setup_logger_kwargs('PelicanAttControllerEnv', None)
    policy_net = policy_core.Policy_Net(S_DIM=S_DIM, A_DIM=A_DIM, EP_MAX=epochs, EP_LEN=episode_steps, GAMMA=discountFactor, LR=I_learningRate, BATCH=minibatch_size, logger_kwargs=logger_kwargs)
    last100Rewards = [0] * 100
    last100RewardsIndex = 0
    last100Filled = False
    all_ep_r = []

    start_time = time.time()

    # start iterating from 'current epoch'.
    for epoch in range(current_epoch + 1, epochs + 1, 1):
        observation = env.reset()
        cumulated_reward = 0
        buffer_s, buffer_a, buffer_r = [], [], []
        # number of timesteps
        for t in range(episode_steps):
            # action = env.action_space.sample()
            # action = ppo.choose_action(np.array(observation))
            action = call_mpc(env, call_nlmpc)
            # print("action: ", action)
            newObservation, reward, done, info = env.step(action)
            buffer_s.append(observation)
            buffer_a.append(action)
            buffer_r.append(reward)
            observation = newObservation

            if (t == episode_steps-1):
                print ("reached the end")
                done = True

            env._flush(force=True)
            cumulated_reward += reward

            # update policy
            if t % minibatch_size == 0 or t == episode_steps or done:
                # v_s_ = policy_net.get_v(np.array(observation))
                bs, ba = np.vstack(buffer_s), np.vstack(buffer_a)
                buffer_s, buffer_a = [], []
                policy_net.update(bs, ba)
            
            if done:
                # print(t)
                break

            # if done:
            #     last100Rewards[last100RewardsIndex] = cumulated_reward
            #     last100RewardsIndex += 1
            #     if last100RewardsIndex >= 100:
            #         last100Filled = True
            #         last100RewardsIndex = 0
            #     m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
            #     h, m = divmod(m, 60)
            #     if not last100Filled:
            #         print("EP " + str(epoch) + " - {} steps".format(t + 1) + " - CReward: " + str(round(cumulated_reward, 4)) + "  Time: %d:%02d:%02d" % (h, m, s))
            #     else:
            #         print("EP " + str(epoch) + " - {} steps".format(t + 1) + " - last100 C_Rewards : " + str(int((sum(last100Rewards) / len(last100Rewards)))) + " - CReward: " + str(round(cumulated_reward, 4)) + "  Eps=" + str(round(explorationRate, 2)) + "  Time: %d:%02d:%02d" % (h, m, s))
            #         # SAVE SIMULATION DATA
            #         if (epoch)%100==0:
            #             #save model weights and monitoring data every 100 epochs.
            #             env._flush()
            #     break
            #     stepCounter += 1

        if epoch % trainning_policy_epochs == 0 and epoch > 0 :
            for i in range(test_policy_epochs):
                observation = env.reset()
                policy_cumulated_reward = 0
                p_buffer_s, p_buffer_a = [], []
                for t in range(episode_steps):
                    action = policy_net.choose_action(np.array(observation))
                    newObservation, reward, done, info = env.step(action)
                    buffer_s.append(observation)
                    buffer_a.append(action)
                    buffer_r.append(reward)
                    observation = newObservation
                    if (t == episode_steps-1):
                        print ("reached the end")
                        done = True
                    env._flush(force=True)
                    policy_cumulated_reward += reward

                    if done:
                        m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                        h, m = divmod(m, 60)
                        print("EP " + str(i) + " - {} test number".format(epoch // trainning_policy_epochs + 1) + " - CReward: " + str(round(policy_cumulated_reward, 4)) + "  Time: %d:%02d:%02d" % (h, m, s))
                        policy_net.logger.store(policy_reward=policy_cumulated_reward)
                        break
            policy_net.logger.log_tabular('policy_reward', average_only=True)
            policy_net.logger.log_tabular('LossPi', average_only=True)
            policy_net.logger.dump_tabular()
            # save model
            policy_net.logger.save_state({}, None)

    env.close()