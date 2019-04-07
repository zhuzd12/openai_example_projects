#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.td3 import core
from spinup.algos.td3.core import get_vars
from openai_ros.task_envs.pelican import pelican_controller
from openai_ros.task_envs.pelican import pelican_controller
from planning_msgs.srv import MotorControllerService

import rospy
import rospkg
import os.path as osp
import joblib
import os
from spinup.utils.logx import EpochLogger
from spinup.utils.logx import restore_tf_graph
from spinup.utils.run_utils import setup_logger_kwargs
import math
from tensorflow.python.tools import inspect_checkpoint as chkp


def load_policy(fpath=None, itr='last', deterministic=False, hidden_sizes=[64,64], activation=tf.nn.tanh, output_activation=None, action_space=None):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    if deterministic and 'mu' in model.keys():
        print('Using deterministic action op.')
        mu = model['mu']
    else:
        print('Using default action op.')
        mu = model['pi']
    
    x = model['x']
    a = model['a']
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high
    # saver = tf.train.Saver()
    with tf.variable_scope('main'):
        # with tf.variable_scope('pi'):
        #     pi = act_limit * core.mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
        # todo: load pi parameters from model after tf.initialize
        # pi = model['pi']
        # saver.restore(sess, osp.join(fpath, 'simple_save'+itr+'/variables')) 

        with tf.variable_scope('q1'):
            q1 = tf.squeeze(core.mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        with tf.variable_scope('q2'):
            q2 = tf.squeeze(core.mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        with tf.variable_scope('q1', reuse=True):
            q1_pi = tf.squeeze(core.mlp(tf.concat([x,model['pi']], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    return sess, model['x'], model['a'],  model['pi'], q1, q2, q1_pi

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
def td3(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=10000, replay_size=int(5e3), gamma=0.99, 
        polyak=0.995, pi_lr=1e-4, q_lr=1e-4, batch_size=64, start_epochs=0, 
        act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2, 
        max_ep_len=500, policy_path=None, logger_kwargs=dict(), save_freq=100, UPDATE_STEP=10, test_freq=100):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    test_logger_kwargs = dict()
    test_logger_kwargs['output_dir'] = osp.join(logger_kwargs['output_dir'], "test")
    test_logger_kwargs['exp_name'] = logger_kwargs['exp_name']
    test_logger = EpochLogger(**test_logger_kwargs)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, do not assumes all dimensions share the same bound!
    act_limit = env.action_space.high
    act_limit_low = env.action_space.low

    act_noise_limit = act_noise*act_limit

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    if policy_path is None:
        # Inputs to computation graph
        x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
        
        # Target policy network
        with tf.variable_scope('target'):
            pi_targ, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)
        
        # Target Q networks
        with tf.variable_scope('target', reuse=True):

            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, act_limit_low, act_limit)

            # Target Q-values, using action from target policy
            _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)
        sess = tf.Session()
    else:
        # sess, x_ph, a_ph, pi, q1, q2, q1_pi = load_policy(fpath=policy_path, itr='last', deterministic=False, **ac_kwargs)
        # x2_ph, r_ph, d_ph = core.placeholders( obs_dim, None, None)

        #  # Target policy network
        #  # todo: copy parameters from main
        # with tf.variable_scope('target'):
        #     pi_targ, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)
        # # Target Q networks
        # with tf.variable_scope('target', reuse=True):

        #     # Target policy smoothing, by adding clipped noise to target actions
        #     epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        #     epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        #     a2 = pi_targ + epsilon
        #     a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        #     # Target Q-values, using action from target policy
        #     _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)
        # Inputs to computation graph
        x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

        # Main outputs from computation graph
        var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main'])
        print('\nNumber of parameters: \t pi: %d, \t total: %d\n'%var_counts)
        with tf.variable_scope('main'):
            pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
        
        # Target policy network
        with tf.variable_scope('target'):
            pi_targ, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)
        
        # Target Q networks
        with tf.variable_scope('target', reuse=True):

            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, act_limit_low, act_limit)

            # Target Q-values, using action from target policy
            _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)
        # sess = tf.Session()

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

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

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # uninitialized_vars = []
    # for var in tf.all_variables():
    #     try:
    #         sess.run(var)
    #     except tf.errors.FailedPreconditionError:
    #         uninitialized_vars.append(var)
    # init_new_vars_op = tf.initialize_variables(uninitialized_vars)

    # todo: reinitialize pi
    pi_ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main/pi')
    # chkp.print_tensors_in_checkpoint_file("/model/model_ckpt", tensor_name='pi', all_tensors=False, all_tensor_names = '')
    saver_pi = tf.train.Saver(pi_ref_vars) 
    # # model = restore_tf_graph(sess, osp.join(policy_path, 'simple_save'))
    # with tf.variable_scope('main', reuse=True):
    saver_pi.restore(sess, './model')
        # pi = model['pi']
        # pass

    sess.run(target_init)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\n updated Number of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q1': q1, 'q2': q2})

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        # todo: add act_limit scale noise
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, act_limit_low, act_limit)
    
    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs
    test_num = 0
    test_policy_epochs = 91
    episode_steps = 500
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(0, epochs):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        # test policy
        if epoch % test_freq == 0:
            env.unwrapped._set_test_mode(True)
            for i in range(test_policy_epochs):
                observation = env.reset()
                policy_cumulated_reward = 0
                for t in range(episode_steps):
                    action = get_action(np.array(observation), act_noise_limit)
                    newObservation, reward, done, info = env.step(action)
                    observation = newObservation
                    if (t == episode_steps-1):
                        print ("reached the end")
                        done = True
                    policy_cumulated_reward += reward

                    if done:
                        test_logger.store(policy_reward=policy_cumulated_reward)
                        test_logger.store(policy_steps=t)
                        # print(info)
                        test_logger.store(arrive_des=info['arrive_des'])
                        break
                    else:
                        pass            
            test_logger.log_tabular('epoch', epoch)
            test_logger.log_tabular('policy_reward', average_only=True)
            test_logger.log_tabular('policy_steps', average_only=True)
            test_logger.log_tabular('arrive_des', average_only=True)
            test_logger.dump_tabular()

        # train policy
        env.unwrapped._set_test_mode(False)
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        for t in range(steps_per_epoch):
            if epoch > start_epochs:
                a = get_action(np.array(o), act_noise_limit)
            else:
                a = env.action_space.sample()

            # Step the env
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            if (t == steps_per_epoch-1):
                print ("reached the end")
                d = True

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            # d = False if ep_len==max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            if d:
                """
                Perform all TD3 updates at the end of the trajectory
                (in accordance with source code of TD3 published by
                original authors).
                """
                for j in range(ep_len):
                    batch = replay_buffer.sample_batch(batch_size)
                    feed_dict = {x_ph: batch['obs1'],
                                x2_ph: batch['obs2'],
                                a_ph: batch['acts'],
                                r_ph: batch['rews'],
                                d_ph: batch['done']
                                }
                    q_step_ops = [q_loss, q1, q2, train_q_op]
                    outs = sess.run(q_step_ops, feed_dict)
                    logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                    if j % policy_delay == 0:
                        # Delayed policy update
                        outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                        logger.store(LossPi=outs[0])

                logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                break

        # End of epoch wrap-up
        if epoch > 0 and (epoch % save_freq == 0) or (epoch == epochs-1):

            # Save model
            logger.save_state({}, None)

            # Test the performance of the deterministic version of the agent.
            test_num += 1
            # test_agent(test_num=test_num)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    rospy.init_node('pelican_attitude_controller_td3_training', anonymous=True, log_level=rospy.WARN)
    default_fpath = osp.join(osp.abspath(osp.pardir),'data/Pelican_motor_controller_dagger_for_ppo/Pelican_motor_controller_dagger_for_ppo_s7')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PelicanNavControllerEnv-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--exp_name', type=str, default='td3')
    parser.add_argument('--activation', type=str, default=tf.nn.tanh)
    parser.add_argument('--output_activation', type=str, default=None)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--fpath', type=str, default=default_fpath)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, DEFAULT_DATA_DIR)
    outdir = '/tmp/openai_ros_experiments/'
    env = gym.make(args.env)
    # env = gym.wrappers.Monitor(env, outdir, force=True)
    td3(lambda : env, actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, activation=args.activation, output_activation=args.output_activation),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,policy_path=args.fpath,
        logger_kwargs=logger_kwargs)