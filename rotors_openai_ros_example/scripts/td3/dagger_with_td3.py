#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.td3 import core
from spinup.algos.td3.core import get_vars
from spinup.utils.logx import EpochLogger
from spinup.utils.logx import colorize
from openai_ros.task_envs.pelican import pelican_attitude_controller
from openai_ros.task_envs.pelican import pelican_controller
from spinup.utils.run_utils import setup_logger_kwargs
from planning_msgs.srv import MotorControllerService
from spinup.utils.logx import restore_tf_graph

import rospy
import rospkg
import os.path as osp

class DaggerReplayBuffer:
    """
    A simple FIFO experience replay buffer for DAGGER agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        # self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.rets_buf = np.zeros(size, dtype=np.float32)
        # self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, ret):
        self.obs1_buf[self.ptr] = obs
        # self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.rets_buf[self.ptr] = ret
        # self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    
    def stores(self, obs, act, rew, ret):
        for i in range(len(obs)):
            self.store(obs[i], act[i], rew[i], ret[i])

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    rets=self.rets_buf[idxs])

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

def call_ref_controller(env, controller_service):
    odo = env.unwrapped.get_odom()
    target_pose = env.unwrapped.get_target_pose()
    try:
        res = controller_service(odo, target_pose)
    except (rospy.ServiceException) as e:
        print("/pelican/call_motor_controller service call failed")
    return [res.cmd.angular_velocities[i] for i in range(env.action_space.shape[-1])]

"""

TD3 (Twin Delayed DDPG)

"""
def td3(env_fn, expert=None, policy_path=None, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=500, epochs=1000, replay_size=int(5e3), gamma=0.99, 
        polyak=0.995, pi_lr=1e-4, q_lr=1e-4, batch_size=64, start_epochs=500,dagger_epochs=500, pretrain_epochs=50,
        dagger_noise=0.02, act_noise=0.02, target_noise=0.02, noise_clip=0.5, policy_delay=2, 
        max_ep_len=500, logger_kwargs=dict(), save_freq=50, UPDATE_STEP=10):
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

    # test_logger_kwargs = dict()
    # test_logger_kwargs['output_dir'] = osp.join(logger_kwargs['output_dir'], "test")
    # test_logger_kwargs['exp_name'] = logger_kwargs['exp_name']
    # test_logger = EpochLogger(**test_logger_kwargs)

    # pretrain_logger_kwargs = dict()
    # pretrain_logger_kwargs['output_dir'] = osp.join(logger_kwargs['output_dir'], "pretrain")
    # pretrain_logger_kwargs['exp_name'] = logger_kwargs['exp_name']
    # pretrain_logger = EpochLogger(**pretrain_logger_kwargs)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, do not assumes all dimensions share the same bound!
    act_limit = env.action_space.high/2
    act_high_limit = env.action_space.high
    act_low_limit = env.action_space.low

    act_noise_limit = act_noise*act_limit
    sess = tf.Session()
    if policy_path is None:
        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = env.action_space

        # Inputs to computation graph
        x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
        tfa_ph = core.placeholder(act_dim)
        
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
            a2 = tf.clip_by_value(a2, act_low_limit, act_high_limit)

            # Target Q-values, using action from target policy
            _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)
    
    else:
        # sess = tf.Session()
        model = restore_tf_graph(sess, osp.join(policy_path, 'simple_save'))
        x_ph, a_ph, x2_ph, r_ph, d_ph = model['x_ph'], model['a_ph'], model['x2_ph'], model['r_ph'], model['d_ph']
        pi, q1, q2, q1_pi = model['pi'], model['q1'], model['q2'], model['q1_pi']
        pi_targ, q1_targ, q2_targ = model['pi_targ'], model['q1_targ'], model['q2_targ']
        tfa_ph = core.placeholder(act_dim)
        dagger_epochs = 0
        

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    dagger_replay_buffer = DaggerReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)
    

    if policy_path is None:
        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_targ)

        # dagger loss
        dagger_pi_loss = tf.reduce_mean(tf.square(pi-tfa_ph))
        # TD3 losses
        pi_loss = -tf.reduce_mean(q1_pi)
        q1_loss = tf.reduce_mean((q1-backup)**2)
        q2_loss = tf.reduce_mean((q2-backup)**2)
        q_loss = tf.add(q1_loss, q2_loss)
        pi_loss = tf.identity(pi_loss, name = "pi_loss")
        q1_loss = tf.identity(q1_loss, name = "q1_loss")
        q2_loss = tf.identity(q2_loss, name = "q2_loss")
        q_loss = tf.identity(q_loss, name = "q_loss")

        # Separate train ops for pi, q
        dagger_pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
        train_dagger_pi_op = dagger_pi_optimizer.minimize(dagger_pi_loss, var_list=get_vars('main/pi'), name='train_dagger_pi_op')
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'), name='train_pi_op')
        train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'), name='train_q_op')

        # Polyak averaging for target variables
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
        sess.run(tf.global_variables_initializer())
    else:
        graph = tf.get_default_graph()
        # opts = graph.get_operations()
        # print (opts)
        pi_loss = model['pi_loss']
        q1_loss = model['q1_loss']
        q2_loss = model['q2_loss']
        q_loss = model['q_loss']
        train_q_op = graph.get_operation_by_name('train_q_op')
        train_pi_op = graph.get_operation_by_name('train_pi_op')
        # target_update = graph.get_operation_by_name('target_update')
        # target_init = graph.get_operation_by_name('target_init')
        # Polyak averaging for target variables
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x_ph': x_ph, 'a_ph': a_ph, 'x2_ph': x2_ph, 'r_ph': r_ph, 'd_ph': d_ph}, \
         outputs={'pi': pi, 'q1': q1, 'q2': q2, 'q1_pi': q1_pi, 'pi_targ': pi_targ, 'q1_targ': q1_targ, 'q2_targ': q2_targ, \
             'pi_loss': pi_loss, 'q1_loss': q1_loss, 'q2_loss': q2_loss, 'q_loss': q_loss})

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        # todo: add act_limit scale noise
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, act_low_limit, act_high_limit)

    def choose_action(s, add_noise=False):
        s = s[np.newaxis, :]
        a = sess.run(pi, {x_ph: s})[0]
        if add_noise:
            noise = dagger_noise * act_high_limit * np.random.normal(size=a.shape)
            a = a + noise
        return np.clip(a, act_low_limit, act_high_limit)

    def test_agent(n=81, test_num=1):
        n = env.unwrapped._set_test_mode(True)
        con_flag = False
        for j in range(n):
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, info = env.step(choose_action(np.array(o), 0))
                ep_ret += r
                ep_len += 1
                if d:
                    test_logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
                    test_logger.store(arrive_des=info['arrive_des'])
                    test_logger.store(arrive_des_appro=info['arrive_des_appro'])
                    if not info['out_of_range']:
                        test_logger.store(converge_dis=info['converge_dis'])
                        con_flag = True
                    test_logger.store(out_of_range=info['out_of_range'])
                    # print(info)
        # test_logger.dump_tabular()
        # time.sleep(10)
        if not con_flag:
            test_logger.store(converge_dis=10000)
        env.unwrapped._set_test_mode(False)


    start_time = time.time()
    env.unwrapped._set_test_mode(False)
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs
    test_num = 0

    total_env_t = 0
    print(colorize("begin dagger training", 'green', bold=True))
    # Main loop for dagger pretrain
    for epoch in range(1, dagger_epochs+1, 1):
        obs, acs, rewards = [], [], []
        # number of timesteps
        for t in range(steps_per_epoch):
            # action = env.action_space.sample()
            # action = ppo.choose_action(np.array(observation))
            obs.append(o)
            ref_action = call_ref_controller(env, expert)
            if(epoch < pretrain_epochs):
                action = ref_action
            else:
                action = choose_action(np.array(o), True)

            o2, r, d, info = env.step(action)
            ep_ret += r
            ep_len += 1
            total_env_t += 1
            
            acs.append(ref_action)
            rewards.append(r)
            # Store experience to replay buffer
            replay_buffer.store(o, action, r, o2, d)
            
            o = o2

            if (t == steps_per_epoch-1):
                # print ("reached the end")
                d = True
            
            if d:
                 # collected data to replaybuffer
                max_step = len(np.array(rewards))
                q = [np.sum(np.power(gamma, np.arange(max_step - t)) * rewards[t:]) for t in range(max_step)]
                dagger_replay_buffer.stores(obs, acs, rewards, q)

                # update policy
                for _ in range(int(max_step/5)):
                    batch = dagger_replay_buffer.sample_batch(batch_size)
                    feed_dict = {x_ph: batch['obs1'], tfa_ph: batch['acts']}
                    q_step_ops = [dagger_pi_loss, train_dagger_pi_op]
                    for j in range(UPDATE_STEP):
                        outs = sess.run(q_step_ops, feed_dict)
                    logger.store(LossPi = outs[0])
                
                # train q function
                for j in range(int(max_step/5)):
                    batch = replay_buffer.sample_batch(batch_size)
                    feed_dict = {x_ph: batch['obs1'],
                                x2_ph: batch['obs2'],
                                a_ph: batch['acts'],
                                r_ph: batch['rews'],
                                d_ph: batch['done']
                                }
                    q_step_ops = [q_loss, q1, q2, train_q_op]
                    # for _ in range(UPDATE_STEP):
                    outs = sess.run(q_step_ops, feed_dict)
                    logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])
                
                    if j % policy_delay == 0:
                            # Delayed target update
                            outs = sess.run([target_update], feed_dict)
                            # logger.store(LossPi=outs[0])
                
                # logger.store(LossQ=1000000, Q1Vals=1000000, Q2Vals=1000000)
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                break 

        # End of epoch wrap-up
        if epoch > 0 and (epoch % save_freq == 0) or (epoch == dagger_epochs):
            # Save model
            logger.save_state({}, None)

            # Test the performance of the deterministic version of the agent.
            test_num += 1
            test_agent(test_num=test_num)

            # Log info about epoch
            test_logger.log_tabular('epoch', epoch)
            test_logger.log_tabular('TestEpRet', average_only=True)
            test_logger.log_tabular('TestEpLen', average_only=True)
            test_logger.log_tabular('arrive_des', average_only=True)
            test_logger.log_tabular('converge_dis', average_only=True)
            test_logger.log_tabular('out_of_range', average_only=True)
            test_logger.dump_tabular()

    sess.run(target_init)
    print(colorize("begin td3 training", 'green', bold=True))
    # Main loop: collect experience in env and update/log each epoch
    # total_env_t = 0
    for epoch in range(1, epochs + 1, 1):

        # End of epoch wrap-up
        if epoch > 0 and (epoch % save_freq == 0) or (epoch == epochs):

            # Save model
            logger.save_state({}, None)

            # Test the performance of the deterministic version of the agent.
            test_num += 1
            test_agent(test_num=test_num)

            # Log info about epoch
            test_logger.log_tabular('epoch', epoch)
            test_logger.log_tabular('TestEpRet', average_only=True)
            test_logger.log_tabular('TestEpLen', average_only=True)
            test_logger.log_tabular('arrive_des', average_only=True)
            test_logger.log_tabular('converge_dis', average_only=True)
            test_logger.log_tabular('out_of_range', average_only=True)
            test_logger.dump_tabular()

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        for t in range(steps_per_epoch):
            if epoch > start_epochs:
                a = get_action(np.array(o), act_noise_limit)
            else:
                a = env.action_space.sample()
                # ref_action = call_ref_controller(env, expert)

            # Step the env
            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            total_env_t += 1
            

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            # d = False if ep_len==max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            if (t == steps_per_epoch-1):
                # print ("reached the end")
                d = True

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
                    # for _ in range(UPDATE_STEP):
                    outs = sess.run(q_step_ops, feed_dict)
                    logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                    if j % policy_delay == 0:
                        # Delayed policy update
                        outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                        logger.store(LossPi=outs[0])

                logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                break

        

if __name__ == '__main__':
    rospy.init_node('pelican_attitude_controller_td3_training', anonymous=True, log_level=rospy.WARN)
    default_fpath =  osp.join(osp.abspath(osp.dirname(__file__)),'data/td3_dagger/td3_dagger_s0')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PelicanNavControllerEnv-v0')
    parser.add_argument('--hid', type=int, default=72)
    parser.add_argument('--l', type=int, default=3)
    parser.add_argument('--activation', type=str, default=tf.tanh)
    parser.add_argument('--output_activation', type=str, default=None)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--dagger_epochs', type=int, default=0)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3_dagger_no_turbulence')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')
    print(DEFAULT_DATA_DIR)
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, DEFAULT_DATA_DIR)
    outdir = '/tmp/openai_ros_experiments/'
    env = gym.make(args.env)
    # env = gym.wrappers.Monitor(env, outdir, force=True)
    ref_controller = rospy.ServiceProxy('/pelican/call_motor_controller', MotorControllerService)
    td3(lambda : env, policy_path=default_fpath, expert=ref_controller, actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, activation=args.activation, output_activation=args.output_activation),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,start_epochs=args.start_epochs, dagger_epochs=args.dagger_epochs, pretrain_epochs=args.pretrain_epochs,
        save_freq=args.save_freq, logger_kwargs=logger_kwargs)