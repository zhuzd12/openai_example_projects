import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import random
import memory
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.logx import EpochLogger

def mlp(x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

class Policy_Net(object):
    def __init__(self, env, EP_MAX=1000, EP_LEN=250, GAMMA=0.99, LR = 0.0001, BATCH=32, UPDATE_STEP=10, hidden_sizes = (64,64), activation=tf.tanh, output_activation=tf.tanh, act_noise_amount=0.01, logger_kwargs=dict()):
        
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.EP_MAX = EP_MAX
        self.EP_LEN = EP_LEN
        self.GAMMA = GAMMA
        self.LR = LR
        self.BATCH = BATCH
        self.UPDATE_STEP = UPDATE_STEP
        self.S_DIM = env.observation_space.shape[-1]
        self.A_DIM = env.action_space.shape[-1]
        self.act_high = env.action_space.high
        self.act_low = env.action_space.low
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.act_noise_amount = act_noise_amount
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
    
        # policy
        self.pi, self.pi_params = self._build_net('pi', trainable=True)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.pi -  self.tfa))
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)
        tf.summary.FileWriter("log/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # Setup model saving
        self.logger.setup_tf_saver(self.sess, inputs={'x': self.tfs, 'a': self.tfa}, outputs={'pi': self.pi})
    
    def _build_net(self, name, trainable):
        with tf.variable_scope(name):
            # l1 = tf.layers.dense(self.tfs, 64, tf.nn.relu, trainable=trainable)
            # l2 = tf.layers.dense(l1, 64, tf.nn.relu, trainable=trainable)
            # p_mu_k =  tf.constant(self.act_high, dtype=tf.float32, shape=(self.A_DIM,))
            # sigma_k =  tf.constant([0.5, 0.3, 0.3, 5.0], dtype=tf.float32, shape=(self.A_DIM,))
            # p_mu = p_mu_k * tf.layers.dense(l1, self.A_DIM, tf.nn.tanh, trainable=trainable)
            # sigma = sigma_k * tf.layers.dense(l1, self.A_DIM, tf.nn.softplus, trainable=trainable)
            # norm_dist = tf.distributions.Normal(loc=mu+deviation, scale=sigma)
            act_limit = self.act_high
            pi = act_limit * mlp(self.tfs, list(self.hidden_sizes)+[self.A_DIM], self.activation, self.output_activation)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return pi, params
    
    def choose_action(self, s, add_noise=False):
        s = s[np.newaxis, :]
        a = self.sess.run(self.pi, {self.tfs: s})[0]
        if add_noise:
            noise = self.act_noise_amount * self.act_high * np.random.normal(size=a.shape)#
            a = a + noise
        return np.clip(a, self.act_low, self.act_high)

    def update(self, s, a):
        # update policy
        for _ in range(self.UPDATE_STEP):
            _, loss = self.sess.run([self.train_op, self.loss], {self.tfs: s, self.tfa: a})
        self.logger.store(LossPi=loss)
    
    def save(self):
        pass
