#!/usr/bin/env python3

import gym
import time
from distutils.dir_util import copy_tree
import os
import json
import single_thread_ppo

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
from openai_ros.task_envs.pelican import pelican_willowgarage
# ROS packages required
import rospy
import rospkg

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        os.unlink(file)

if __name__ == '__main__':
    rospy.init_node('pelican_ppo_controller_training', anonymous=True, log_level=rospy.WARN)
    env = gym.make('PelicanNavWillowgarageEnv-v0')
    rospy.loginfo("Gym environment done")
    outdir = '/tmp/openai_ros_experiments/'

    continue_execution = False
    #fill this if continue_execution=True
    weights_path = '/tmp/pelican_willowgarage_ppo_ep200.h5'
    monitor_path = '/tmp/pelican_willowgarag_ppo_ep200'
    params_json  = '/tmp/pelican_willowgarag_ppo_ep200.json'

    A_DIM = env.unwrapped.a_dim
    S_DIM = env.unwrapped.s_dim
    epochs = 100000
    episode_steps = 500
    propeller_hovering_speed = rospy.get_param("/pelican/propeller_hovering_speed")

    if not continue_execution:
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
    ppo = single_thread_ppo.PPO(S_DIM=S_DIM, A_DIM=A_DIM, EP_MAX=epochs, EP_LEN=episode_steps, GAMMA=discountFactor, A_LR=A_learningRate, C_LR=C_learningRate,BATCH=minibatch_size, propeller_hovering_speed=propeller_hovering_speed)
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

            #action = env.action_space.sample()
            action = ppo.choose_action(np.array(observation))
            #print("action: ", action)
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

            # update ppo
            if t % minibatch_size == 0 or t == episode_steps or done:
                v_s_ = ppo.get_v(np.array(observation))
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + discountFactor * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)

            if done:
                last100Rewards[last100RewardsIndex] = cumulated_reward
                last100RewardsIndex += 1
                if last100RewardsIndex >= 100:
                    last100Filled = True
                    last100RewardsIndex = 0
                m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                h, m = divmod(m, 60)
                if not last100Filled:
                    print("EP " + str(epoch) + " - {} steps".format(t + 1) + " - CReward: " + str(round(cumulated_reward, 2)) + "  Time: %d:%02d:%02d" % (h, m, s))
                else:
                    print("EP " + str(epoch) + " - {} steps".format(t + 1) + " - last100 C_Rewards : " + str(int((sum(last100Rewards) / len(last100Rewards)))) + " - CReward: " + str(round(cumulated_reward, 2)) + "  Eps=" + str(round(explorationRate, 2)) + "  Time: %d:%02d:%02d" % (h, m, s))
                    # SAVE SIMULATION DATA
                    if (epoch)%100==0:
                        #save model weights and monitoring data every 100 epochs.
                        env._flush()
                break
                stepCounter += 1

        if epoch == 1:
            all_ep_r.append(cumulated_reward)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + cumulated_reward * 0.1)

    env.close()