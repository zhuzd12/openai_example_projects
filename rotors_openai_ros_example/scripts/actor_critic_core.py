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
import spinup.algos.ppo.core as core

def mlp(x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

class AC_Net(object):
    def __init__(self, env, EP_MAX=1000, EP_LEN=500, GAMMA=0.99, AR = 0.0001, CR = 0.0001, BATCH=32, UPDATE_STEP=10, hidden_sizes = (64,64), activation=tf.tanh, output_activation=tf.tanh, act_noise_amount=0.01, logger_kwargs=dict()):
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.EP_MAX = EP_MAX
        self.EP_LEN = EP_LEN
        self.GAMMA = GAMMA
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
        # self.sess = tf.Session()
        self.PROJECT_ROOT = logger_kwargs['output_dir']
        self.save_times = 0
        # self.sess = sess
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
    
        # policy and value 
        self.tfr = tf.placeholder(tf.float32, [None,], 'reward_to_go')
        # ac_kwargs['action_space'] = env.action_space
        # ac_kwargs['hidden_sizes'] = (64,64)
        # self.pi, self.logp, self.logp_pi, self.v = core.mlp_actor_critic(self.tfs, self.tfa, **ac_kwargs)
        self.pi, _ = self._build_net('pi', trainable=True)
        with tf.variable_scope('v'):
            self.v = tf.squeeze(mlp(self.tfs, list(self.hidden_sizes)+[1], self.activation, None), axis=1)

        # with tf.variable_scope('q1'):
        #     q1 = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

        with tf.variable_scope('loss'):
            self.pi_loss = tf.reduce_mean(tf.square(self.pi -  self.tfa))
            self.v_loss = tf.reduce_mean((self.tfr - self.v)**2)
            # self.q_loss = tf.reduce_mean((self.tfr - self.v)**2)

        with tf.variable_scope('train'):
            self.train_pi = tf.train.AdamOptimizer(AR).minimize(self.pi_loss)
            self.train_v = tf.train.AdamOptimizer(CR).minimize(self.v_loss)


        pi_ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main/pi')
        self.saver = tf.train.Saver(pi_ref_vars)
        
        self.sess = tf.Session()
        tf.summary.FileWriter("log/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # # Setup model saving
        self.logger.setup_tf_saver(self.sess, inputs={'x': self.tfs, 'a': self.tfa}, outputs={'pi': self.pi, 'v': self.v})
    
    def _build_net(self, name, trainable):
        with tf.variable_scope(name):
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
            _, loss = self.sess.run([self.train_pi, self.pi_loss], {self.tfs: s, self.tfa: a})
        self.logger.store(LossPi=loss)
    
    def update_v(self, s, r):
        # update value function
        for _ in range(self.UPDATE_STEP):
            _, loss = self.sess.run([self.train_v, self.v_loss], {self.tfs: s, self.tfr: r})
        self.logger.store(LossV=loss)
    
    # def update_q(self, s, a, r):
    #     # update value function
    #     for _ in range(self.UPDATE_STEP):
    #         _, loss = self.sess.run([self.train_q, self.q_loss], {self.tfs: s, self.tfa: a, self.tfr: r})
    #     self.logger.store(LossQ=loss)
    
    def save(self):
        # pass
        self.saver.save(self.sess, os.path.join(self.PROJECT_ROOT, "model/model.ckpt"), global_step=self.save_times)
        self.save_times = self.save_times + 1
