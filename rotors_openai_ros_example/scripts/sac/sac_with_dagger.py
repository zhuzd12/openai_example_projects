import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sac import core
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from openai_ros.task_envs.pelican import pelican_attitude_controller
from openai_ros.task_envs.pelican import pelican_controller
from spinup.utils.logx import colorize

import rospy
import rospkg
from planning_msgs.srv import MotorControllerService

import joblib
import os
import os.path as osp
from spinup.utils.logx import EpochLogger
from spinup.utils.logx import restore_tf_graph
from spinup.utils.run_utils import setup_logger_kwargs
import math

def call_ref_controller(env, controller_service):
    odo = env.unwrapped.get_odom()
    target_pose = env.unwrapped.get_target_pose()
    try:
        res = controller_service(odo, target_pose)
    except (rospy.ServiceException) as e:
        print("/pelican/call_motor_controller service call failed")
    return [res.cmd.angular_velocities[i] for i in range(env.action_space.shape[-1])]

class DaggerReplayBuffer:
    """
    A simple FIFO experience replay buffer for DAGGER agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        # self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        # self.rets_buf = np.zeros(size, dtype=np.float32)
        # self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew):
        self.obs1_buf[self.ptr] = obs
        # self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        # self.rets_buf[self.ptr] = ret
        # self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    
    def stores(self, obs, act, rew):
        for i in range(len(obs)):
            self.store(obs[i], act[i], rew[i])

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs])

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
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

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""

"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
def sac(env_fn,  expert=None, policy_path=None, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=500, epochs=100000, replay_size=int(5e3), gamma=0.99, 
        dagger_noise=0.02, polyak=0.995, lr=1e-4, alpha=0.2, batch_size=64, dagger_epochs=200, pretrain_epochs=50,
        max_ep_len=500, logger_kwargs=dict(), save_freq=50, update_steps=10):
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
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. 
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

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

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

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

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(obs_dim)
    print(act_dim)
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    act_high_limit = env.action_space.high
    act_low_limit = env.action_space.low

    sess = tf.Session()
    if policy_path is None:
        # Inputs to computation graph
        x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
        tfa_ph = core.placeholder(act_dim)
        # Main outputs from computation graph
        with tf.variable_scope('main'):
            mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)
        
        # Target value network
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, v_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)
        # sess.run(tf.global_variables_initializer())
    
    else:
        # load pretrained model
        model = restore_tf_graph(sess, osp.join(policy_path, 'simple_save'))
        x_ph, a_ph, x2_ph, r_ph, d_ph = model['x_ph'], model['a_ph'], model['x2_ph'], model['r_ph'], model['d_ph']
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = model['mu'], model['pi'], model['logp_pi'], model['q1'], model['q2'], model['q1_pi'], model['q2_pi'], model['v']
        # tfa_ph = core.placeholder(act_dim)
        tfa_ph = model['tfa_ph']

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    dagger_replay_buffer = DaggerReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)


    # print(obs_dim)
    # print(act_dim)

    # SAC objectives
    if policy_path is None:
        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
        v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

        # Soft actor-critic losses
        dagger_pi_loss = tf.reduce_mean(tf.square(mu-tfa_ph))
        pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
        value_loss = q1_loss + q2_loss + v_loss

        # Policy train op 
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        dagger_pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_dagger_pi_op = dagger_pi_optimizer.minimize(dagger_pi_loss, name='train_dagger_pi_op')

        pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'), name='train_pi_op')
        # sess.run(tf.variables_initializer(pi_optimizer.variables()))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        value_params = get_vars('main/q') + get_vars('main/v')
        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(value_loss, var_list=value_params, name='train_value_op')
            # sess.run(tf.variables_initializer(value_optimizer.variables()))

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                    for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi, 
                    train_pi_op, train_value_op, target_update]

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
        sess.run(tf.global_variables_initializer())
    else:
        graph = tf.get_default_graph()
        dagger_pi_loss = model['dagger_pi_loss']
        pi_loss = model['pi_loss']
        q1_loss = model['q1_loss']
        q2_loss = model['q2_loss']        
        v_loss = model['v_loss']

        train_dagger_pi_op = graph.get_operation_by_name('train_dagger_pi_op')
        train_value_op = graph.get_operation_by_name('train_value_op')
        train_pi_op = graph.get_operation_by_name('train_pi_op')
        
        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                    for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi, 
                    train_pi_op, train_value_op, target_update]

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    dagger_step_ops = [q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi, train_value_op, target_update]
    tf.summary.FileWriter("log/", sess.graph)
    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x_ph': x_ph, 'a_ph': a_ph, 'tfa_ph': tfa_ph, 'x2_ph': x2_ph, 'r_ph': r_ph, 'd_ph': d_ph}, \
        outputs={'mu': mu, 'pi': pi, 'v': v, 'logp_pi': logp_pi, 'q1': q1, 'q2': q2, 'q1_pi': q1_pi, 'q2_pi': q2_pi, \
            'pi_loss': pi_loss, 'v_loss': v_loss, 'dagger_pi_loss': dagger_pi_loss, 'q1_loss': q1_loss, 'q2_loss': q2_loss})
    
    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        a = sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]
        return np.clip(a, act_low_limit, act_high_limit)

    def choose_action(s, add_noise=False):
        s = s[np.newaxis, :]
        a = sess.run(mu, {x_ph: s})[0]
        if add_noise:
            noise = dagger_noise * act_high_limit * np.random.normal(size=a.shape)
            a = a + noise
        return np.clip(a, act_low_limit, act_high_limit)

    def test_agent(n=81, test_num=1):
        n = env.unwrapped._set_test_mode(True)
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
                    test_logger.store(converge_dis=info['converge_dis'])
                    test_logger.store(out_of_range=info['out_of_range'])
                    # print(info)
        # test_logger.dump_tabular()
        env.unwrapped._set_test_mode(False)

    def ref_test_agent(n=81, test_num=1):
        n = env.unwrapped._set_test_mode(True)
        for j in range(n):
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a  = call_ref_controller(env, expert)
                o, r, d, info = env.step(a)
                ep_ret += r
                ep_len += 1

                if d:
                    test_logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
                    test_logger.store(arrive_des=info['arrive_des'])
                    test_logger.store(converge_dis=info['converge_dis'])
                    test_logger.store(out_of_range=info['out_of_range'])
                    # print(info)
        # test_logger.dump_tabular()
        env.unwrapped._set_test_mode(False)
    ref_test_agent(test_num = -1)
    test_logger.log_tabular('epoch', -1)
    test_logger.log_tabular('TestEpRet', average_only=True)
    test_logger.log_tabular('TestEpLen', average_only=True)
    test_logger.log_tabular('arrive_des', average_only=True)
    test_logger.log_tabular('converge_dis', average_only=True)
    test_logger.log_tabular('out_of_range', average_only=True)
    test_logger.dump_tabular()


    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    episode_steps = 500
    total_env_t = 0
    test_num = 0
    print(colorize("begin dagger training", 'green', bold=True))
    for epoch in range(1, dagger_epochs + 1, 1):
        # test policy
        if epoch > 0 and (epoch % save_freq == 0) or (epoch == epochs):
            # Save model
            logger.save_state({}, None)
            
            # Test the performance of the deterministic version of the agent.
            test_num += 1
            test_agent(test_num=test_num)
            
            test_logger.log_tabular('epoch', epoch)
            test_logger.log_tabular('TestEpRet', average_only=True)
            test_logger.log_tabular('TestEpLen', average_only=True)
            test_logger.log_tabular('arrive_des', average_only=True)
            test_logger.log_tabular('converge_dis', average_only=True)
            test_logger.log_tabular('out_of_range', average_only=True)
            test_logger.dump_tabular()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True) 
            logger.log_tabular('Q2Vals', with_min_and_max=True) 
            logger.log_tabular('VVals', with_min_and_max=True) 
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

        # train policy
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        env.unwrapped._set_test_mode(False)
        obs, acs, rewards = [], [], []
        for t in range(steps_per_epoch):
            obs.append(o)
            ref_action = call_ref_controller(env, expert)
            if(epoch < pretrain_epochs):
                action = ref_action
            else:
                action = choose_action(np.array(o), True)
            
            o2, r, d, _ = env.step(action)
            o = o2
            acs.append(ref_action)
            rewards.append(r)

            if (t == steps_per_epoch-1):
                print ("reached the end")
                d = True

            # Store experience to replay buffer
            replay_buffer.store(o, action, r, o2, d)

            ep_ret += r
            ep_len += 1
            total_env_t += 1

            if d:
                # Perform partical sac update!
                for j in range(ep_len):
                    batch = replay_buffer.sample_batch(batch_size)
                    feed_dict = {x_ph: batch['obs1'],
                                x2_ph: batch['obs2'],
                                a_ph: batch['acts'],
                                r_ph: batch['rews'],
                                d_ph: batch['done'],
                                }
                    outs = sess.run(dagger_step_ops, feed_dict)
                    logger.store(LossQ1=outs[0], LossQ2=outs[1],
                                LossV=outs[2], Q1Vals=outs[3], Q2Vals=outs[4],
                                VVals=outs[5], LogPi=outs[6])

                # Perform dagger policy update
                dagger_replay_buffer.stores(obs, acs, rewards)
                for _ in range(int(ep_len/5)):
                    batch = dagger_replay_buffer.sample_batch(batch_size)
                    feed_dict = {x_ph: batch['obs1'], tfa_ph: batch['acts']}
                    q_step_ops = [dagger_pi_loss, train_dagger_pi_op]
                    for j in range(10):
                        outs = sess.run(q_step_ops, feed_dict)
                    logger.store(LossPi = outs[0])

                logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                break

    # Main loop: collect experience in env and update/log each epoch
    print(colorize("begin sac training", 'green', bold=True))
    for epoch in range(1, epochs + 1, 1):
        # test policy
        if epoch > 0 and (epoch % save_freq == 0) or (epoch == epochs):
            # Save model
            logger.save_state({}, None)
            
            # Test the performance of the deterministic version of the agent.
            test_num += 1
            test_agent(test_num=test_num)
            
            test_logger.log_tabular('epoch', epoch)
            test_logger.log_tabular('TestEpRet', average_only=True)
            test_logger.log_tabular('TestEpLen', average_only=True)
            test_logger.log_tabular('arrive_des', average_only=True)
            test_logger.log_tabular('converge_dis', average_only=True)
            test_logger.log_tabular('out_of_range', average_only=True)
            test_logger.dump_tabular()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

        # train policy
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        env.unwrapped._set_test_mode(False)
        for t in range(steps_per_epoch):
            a = get_action(np.array(o))

            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            if (t == steps_per_epoch-1):
                print ("reached the end")
                d = True

            replay_buffer.store(o, a, r, o2, d)
            o = o2
            if d:
                """
                Perform all SAC updates at the end of the trajectory.
                This is a slight difference from the SAC specified in the
                original paper.
                """
                for j in range(ep_len):
                    batch = replay_buffer.sample_batch(batch_size)
                    feed_dict = {x_ph: batch['obs1'],
                                x2_ph: batch['obs2'],
                                a_ph: batch['acts'],
                                r_ph: batch['rews'],
                                d_ph: batch['done'],
                                }
                    outs = sess.run(step_ops, feed_dict)
                    logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                                LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                                VVals=outs[6], LogPi=outs[7])

                logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        
if __name__ == '__main__':
    rospy.init_node('pelican_attitude_controller_sac_training', anonymous=True, log_level=rospy.WARN)
    import argparse
    # default_fpath = osp.join(osp.abspath(osp.pardir),'data/Pelican_position_controller_dagger_for_ppo/Pelican_position_controller_dagger_for_ppo_s3')
    default_fpath =  osp.join(osp.abspath(osp.dirname(__file__)),'data/sac_dagger/sac_dagger_s0')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PelicanNavControllerEnv-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--dagger_epochs', type=int, default=0)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--activation', type=str, default=tf.nn.tanh)
    parser.add_argument('--output_activation', type=str, default=tf.nn.tanh)
    parser.add_argument('--exp_name', type=str, default='sac_dagger')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--fpath', type=str, default=default_fpath)
    args = parser.parse_args()

    # mpi_fork(args.cpu)  # run parallel code with mpi
    from spinup.utils.run_utils import setup_logger_kwargs
    DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, DEFAULT_DATA_DIR)

    outdir = '/tmp/openai_ros_experiments/'
    env = gym.make(args.env)
    ref_controller = rospy.ServiceProxy('/pelican/call_motor_controller', MotorControllerService)

    sac(lambda : env, policy_path=default_fpath, expert=ref_controller, actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, activation=args.activation, output_activation=args.output_activation), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        dagger_epochs=args.dagger_epochs, pretrain_epochs=args.pretrain_epochs, save_freq=args.save_freq, logger_kwargs=logger_kwargs)