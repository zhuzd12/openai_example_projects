import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.td3 import core
from spinup.algos.td3.core import get_vars
from spinup.utils.logx import EpochLogger
from openai_ros.task_envs.pelican import pelican_attitude_controller
import rospy
import rospkg

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
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

"""

TD3 (Twin Delayed DDPG)

"""
class td3_rl(object):
    def __init__(self, obs_space, act_space, EP_MAX=1000, EP_LEN=250, GAMMA=0.99, A_LR = 0.0001, C_LR=0.0001, BATCH=32, UPDATE_STEP=10, logger_kwargs=dict()):
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.EP_MAX = EP_MAX
        self.EP_LEN = EP_LEN
        self.GAMMA = GAMMA
        self.LR = LR
        self.BATCH = BATCH
        self.UPDATE_STEP = UPDATE_STEP
        self.obs_space = obs_space
        self.act_space = act_space
        self.obs_dim = self.obs_space.shape[0]
        self.act_dim = self.act_space.shape[0]
        self.act_limit = act_space.high
        
        self.sess = tf.Session()
        # self.tfs = tf.placeholder(tf.float32, [None, self.obs_dim], 'state')
        # self.tfa = tf.placeholder(tf.float32, [None, self.act_dim], 'action')

        tf.set_random_seed(seed)
        np.random.seed(seed)

        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.pi, self.q1, self.q2, self.q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
        
        # Target policy network
        with tf.variable_scope('target'):
            self.pi_targ, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)
        
        # Target Q networks
        with tf.variable_scope('target', reuse=True):

            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, -act_limit, act_limit)

            # Target Q-values, using action from target policy
            _, self.q1_targ, self.q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)

        # # Experience buffer
        # self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_targ)

        # TD3 losses
        pi_loss = -tf.reduce_mean(q1_pi)
        q1_loss = tf.reduce_mean((q1-backup)**2)
        q2_loss = tf.reduce_mean((q2-backup)**2)
        q_loss = q1_loss + q2_loss

        # Separate train ops for pi, q
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
        train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

        # Polyak averaging for target variables
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
