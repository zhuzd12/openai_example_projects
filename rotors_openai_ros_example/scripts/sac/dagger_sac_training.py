import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sac import core
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger
from openai_ros.task_envs.pelican import pelican_attitude_controller
from openai_ros.task_envs.pelican import pelican_controller

import rospy
import rospkg
import os.path as osp
import math
import joblib
import os
from spinup.utils.logx import restore_tf_graph
from spinup.utils.run_utils import setup_logger_kwargs


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

def load_policy(fpath, itr='last', deterministic=False, act_high=1, hidden_sizes=(64,64), activation=tf.tanh):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    print("itr:", itr)
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    if deterministic and 'mu' in model.keys():
        print('Using deterministic action op.')
        with tf.variable_scope("pi", reuse=True):
            mu = model['mu']
    else:
        print('Using default action op.')
        with tf.variable_scope("pi", reuse=True):
            mu = model['pi']
    
    x = model['x']
    a = model['a']

    vf_mlp = lambda x : tf.squeeze(core.mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(tf.concat([x,pi], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([x,a], axis=-1))
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(tf.concat([x,pi], axis=-1))
    sess.run(tf.global_variables_initializer())

    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    act_dim = a.shape.as_list()[-1]
    with tf.variable_scope("pi", reuse=True):
        # log_std = tf.constant(0.01*act_high, dtype=tf.float32, shape=(act_dim,))
        net = core.mlp(x, list(hidden_sizes), activation, activation)
        log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    # log_std = tf.get_variable(name='log_std', initializer=math.log(0.01*act_high[0])*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    with tf.variable_scope("pi", reuse=True):
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_pi = core.gaussian_likelihood(pi, mu, log_std)
    
    

    if 'v' in model.keys():
        print("value function already in model")
        with tf.variable_scope('v'):
            v = model['v']
    else:
        with tf.variable_scope('v'):
            v = vf_mlp(x)

    # get_action = lambda x : sess.run(mu, feed_dict={model['x']: x[None,:]})[0]
    sess.run(tf.initialize_variables([log_std]))

    return sess, model['x'], model['a'], mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v

"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=500, epochs=100000, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-4, alpha=0.2, batch_size=100, start_epochs=1000, 
        max_ep_len=500, policy_path=None, logger_kwargs=dict(), save_freq=100, update_steps=10):
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

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['output_activation'] = None

    if policy_path is None:
        # Inputs to computation graph
        x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)
        
        # Target value network
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, v_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)

    else:
        # todo
         # load pretrained model
        with tf.variable_scope('main'):
            sess, x_ph, a_ph, mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = load_policy(policy_path, itr='last', deterministic=False, act_high=env.action_space.high)
        
        x2_ph, r_ph, d_ph = core.placeholders(None, None, None)

         # Target value network
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, v_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
    value_loss = q1_loss + q2_loss + v_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)

    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    sess.run(tf.variables_initializer(pi_optimizer.variables()))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('main/v')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)
        sess.run(tf.variables_initializer(value_optimizer.variables()))

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
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
                                outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def test_agent(n=91, test_num=1):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        env.unwrapped._set_test_mode(True)
        for i in range(n):
            observation = env.reset()
            policy_cumulated_reward = 0
            for t in range(episode_steps):
                newObservation, reward, done, info = env.step(get_action(np.array(observation), True))
                observation = newObservation
                if (t == episode_steps-1):
                    print ("reached the end")
                    done = True
                policy_cumulated_reward += reward

                if done:
                    test_logger.store(policy_reward=policy_cumulated_reward)
                    test_logger.store(policy_steps=t)
                    test_logger.store(arrive_des=info['arrive_des'])
                    break
                else:
                    pass            
        test_logger.log_tabular('epoch', epoch)
        test_logger.log_tabular('policy_reward', average_only=True)
        test_logger.log_tabular('policy_steps', average_only=True)
        test_logger.log_tabular('arrive_des', average_only=True)
        test_logger.dump_tabular()

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs
    test_num = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        for t in range(steps_per_epoch):
            if epoch > start_epochs:
                a = get_action(np.array(o))
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
                break


        # End of epoch wrap-up
        if epoch > 0 and (epoch % save_freq == 0) or (epoch == epochs-1):
            # Save model
            logger.save_state({}, None)

            # Test the performance of the deterministic version of the agent.
            test_num += 1
            test_agent(test_num=test_num)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
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

if __name__ == '__main__':
    rospy.init_node('pelican_motor_controller_sac_training', anonymous=True, log_level=rospy.WARN)
    import argparse
    default_fpath = osp.join(osp.abspath(osp.pardir),'data/Pelican_motor_controller_dagger_for_ppo/Pelican_motor_controller_dagger_for_ppo_s0')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PelicanNavControllerEnv-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--fpath', type=str, default=default_fpath)
    parser.add_argument('--exp_name', type=str, default='sac_motor_controller')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, DEFAULT_DATA_DIR)
    outdir = '/tmp/openai_ros_experiments/'
    env = gym.make(args.env)
    # env = gym.wrappers.Monitor(env, outdir, force=True)
    sac(lambda : env, actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, policy_path=args.fpath,
        logger_kwargs=logger_kwargs)