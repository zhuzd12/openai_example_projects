"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import memory
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class PPO(object):
    def __init__(self, S_DIM, A_DIM, EP_MAX=1000, EP_LEN=250, GAMMA=0.9, A_LR = 0.0001, C_LR=0.0002, BATCH=32, A_UPDATE_STEP=10, C_UPDATE_STEP=10, propeller_hovering_speed=500.0):
        self.EP_MAX = EP_MAX
        self.EP_LEN = EP_LEN
        self.GAMMA = GAMMA
        self.A_LR = A_LR
        self.C_LR = C_LR
        self.BATCH = BATCH
        self.A_UPDATE_STEP = A_UPDATE_STEP
        self.C_UPDATE_STEP = C_UPDATE_STEP
        self.S_DIM = S_DIM
        self.A_DIM = A_DIM
        self.propeller_hovering_speed = propeller_hovering_speed
        self.METHOD = [
            dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
            dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
        ][1]  # choose the method for optimization

        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / (1e-5 + oldpi.prob(self.tfa))
                surr = ratio * self.tfadv
            if self.METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:  # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1. - self.METHOD['epsilon'], 1. + self.METHOD['epsilon']) * self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if self.METHOD['name'] == 'kl_pen':
            for _ in range(self.A_UPDATE_STEP):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: self.METHOD['lam']})
                if kl > 4*self.METHOD['kl_target']:  # this is in google's paper
                    break
            if kl < self.METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                self.METHOD['lam'] /= 2
            elif kl > self.METHOD['kl_target'] * 1.5:
                self.METHOD['lam'] *= 2
            self.METHOD['lam'] = np.clip(self.METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.A_UPDATE_STEP)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(self.C_UPDATE_STEP)]


    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 50, tf.nn.relu, trainable=trainable)
            deviation = tf.constant(self.propeller_hovering_speed, dtype=tf.float32, shape=(self.A_DIM,))
            p_mu_k =  tf.constant([5.0, 1.57, 1.57], dtype=tf.float32, shape=(self.A_DIM-1,))
            sigma_k =  tf.constant([0.5, 0.3, 0.3, 5.0], dtype=tf.float32, shape=(self.A_DIM,))
            p_mu = p_mu_k * tf.layers.dense(l1, self.A_DIM-1, tf.nn.tanh, trainable=trainable)
            thrust_layer = 20.0 * tf.layers.dense(l1, 1, tf.nn.sigmoid, trainable=trainable)
            mu = tf.concat([p_mu, thrust_layer], axis = 1)
            sigma = sigma_k * tf.layers.dense(l1, self.A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu+deviation, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, [-5.0, -1.57, -1.57, 0.0], [5.0, 1.57, 1.57, 20])

        # select the action with the highest Q value
    def selectAction(self, s, explorationRate, env):
        rand = random.random()
        if rand < explorationRate:
            action = env.action_space.sample()
        else:
            action = self.choose_action(s)
        return action

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]