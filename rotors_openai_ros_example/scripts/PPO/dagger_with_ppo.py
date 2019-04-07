import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.ppo.core as core
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

def load_policy(fpath, itr='last', deterministic=False, act_high=1):

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
    # log_std = tf.constant(0.01*act_high, dtype=tf.float32, shape=(act_dim,))
    log_std = tf.get_variable(name='log_std', initializer=math.log(0.01*act_high[0])*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    with tf.variable_scope("pi", reuse=True):
        pi = mu + tf.random_normal(tf.shape(mu)) * std
    with tf.variable_scope("log_pi"):
        logp = core.gaussian_likelihood(a, mu, log_std)
        logp_pi = core.gaussian_likelihood(pi, mu, log_std)

    if 'v' in model.keys():
        print("value function already in model")
        v = model['v']
    else:
        _, _, _, v = core.mlp_actor_critic(x, a, **ac_kwargs)

    # get_action = lambda x : sess.run(mu, feed_dict={model['x']: x[None,:]})[0]
    sess.run(tf.initialize_variables([log_std]))

    return sess, model['x'], model['a'], mu, pi, logp, logp_pi, v

    # # make function for producing an action given a single state
    # get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # # try to load environment from save
    # # (sometimes this will fail because the environment could not be pickled)

    # return get_action

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
        self.rets_buf = np.zeros(size, dtype=np.float32)
        # self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew):
        self.obs1_buf[self.ptr] = obs
        # self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
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

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]


"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""
def ppo(env_fn, expert=None, policy_path=None, actor_critic=core.mlp_actor_critic_m, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=10000, dagger_epochs=500, pretrain_epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4,
        dagger_noise=0.01, batch_size=64, replay_size=int(5e3), vf_lr=1e-4, train_pi_iters=80, train_v_iters=80, lam=0.999, max_ep_len=500,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, test_freq=10):
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
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        policy_path (str): path of pretrained policy model
            train from scratch if None

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
    test_logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    act_high_limit = env.action_space.high
    act_low_limit = env.action_space.low

    sess = tf.Session()
    if policy_path is None:
        # Inputs to computation graph
        x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
        adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)
        tfa_ph = core.placeholder(act_dim)

        # Main outputs from computation graph
        mu, pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)
        sess.run(tf.global_variables_initializer())
    
    else:
        # load pretrained model
        # sess, x_ph, a_ph, mu, pi, logp, logp_pi, v = load_policy(policy_path, itr='last', deterministic=False, act_high=env.action_space.high)
        # # get_action_2 = lambda x : sess.run(mu, feed_dict={x_ph: x[None,:]})[0]
        # adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)
        model = restore_tf_graph(sess, osp.join(policy_path, 'simple_save'))
        x_ph, a_ph, adv_ph, ret_ph, logp_old_ph = model['x_ph'], model['a_ph'], model['adv_ph'], model['ret_ph'], model['logp_old_ph']
        mu, pi, logp, logp_pi, v = model['mu'], model['pi'], model['logp'], model['logp_pi'], model['v']
        # tfa_ph = core.placeholder(act_dim)
        tfa_ph = model['tfa_ph']

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    print("---------------", local_steps_per_epoch)
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    # print(obs_dim)
    # print(act_dim)
    dagger_replay_buffer = DaggerReplayBuffer(obs_dim=obs_dim[0], act_dim=act_dim[0], size=replay_size)
    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # PPO objectives
    if policy_path is None:
        ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
        v_loss = tf.reduce_mean((ret_ph - v)**2)
        dagger_pi_loss = tf.reduce_mean(tf.square(mu-tfa_ph))

        # Info (useful to watch during learning)
        approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
        approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
        clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # Optimizers
        dagger_pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        optimizer_pi = tf.train.AdamOptimizer(learning_rate=pi_lr)
        optimizer_v = tf.train.AdamOptimizer(learning_rate=vf_lr)
        train_dagger_pi_op = dagger_pi_optimizer.minimize(dagger_pi_loss, name='train_dagger_pi_op')
        train_pi =optimizer_pi.minimize(pi_loss, name='train_pi_op')
        train_v = optimizer_v.minimize(v_loss, name='train_v_op')

        sess.run(tf.variables_initializer(optimizer_pi.variables()))
        sess.run(tf.variables_initializer(optimizer_v.variables()))
        sess.run(tf.variables_initializer(dagger_pi_optimizer.variables()))
    else:
        graph = tf.get_default_graph()
        dagger_pi_loss = model['dagger_pi_loss']
        pi_loss = model['pi_loss']
        v_loss = model['v_loss']
        approx_ent = model['approx_ent']
        approx_kl = model['approx_kl']
        clipfrac = model['clipfrac']

        train_dagger_pi_op = graph.get_operation_by_name('train_dagger_pi_op')
        train_pi = graph.get_operation_by_name('train_pi_op')
        train_v = graph.get_operation_by_name('train_v_op')
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    # Sync params across processes
    # sess.run(sync_all_params())

    tf.summary.FileWriter("log/", sess.graph)
    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x_ph': x_ph, 'a_ph': a_ph, 'tfa_ph': tfa_ph, 'adv_ph': adv_ph, 'ret_ph': ret_ph, 'logp_old_ph': logp_old_ph}, \
        outputs={'mu': mu, 'pi': pi, 'v': v, 'logp': logp, 'logp_pi': logp_pi, 'clipfrac': clipfrac, 'approx_kl': approx_kl, \
            'pi_loss': pi_loss, 'v_loss': v_loss, 'dagger_pi_loss': dagger_pi_loss, 'approx_ent': approx_ent})

    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))
    
    def choose_action(s, add_noise=False):
        s = s[np.newaxis, :]
        a = sess.run(mu, {x_ph: s})[0]
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

    def ref_test_agent(n=81, test_num=1):
        n = env.unwrapped._set_test_mode(True)
        con_flag = False
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
                    test_logger.store(arrive_des_appro=info['arrive_des_appro'])
                    if not info['out_of_range']:
                        test_logger.store(converge_dis=info['converge_dis'])
                        con_flag = True
                    test_logger.store(out_of_range=info['out_of_range'])
                    # print(info)
        # test_logger.dump_tabular()
        if not con_flag:
            test_logger.store(arrive_des=10000)
        env.unwrapped._set_test_mode(False)

    ref_test_agent(test_num = -1)
    test_logger.log_tabular('epoch', -1)
    test_logger.log_tabular('TestEpRet', average_only=True)
    test_logger.log_tabular('TestEpLen', average_only=True)
    test_logger.log_tabular('arrive_des', average_only=True)
    test_logger.log_tabular('arrive_des_appro', average_only=True)
    test_logger.log_tabular('converge_dis', average_only=True)
    test_logger.log_tabular('out_of_range', average_only=True)
    test_logger.dump_tabular()

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    test_policy_epochs = 91
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
            test_logger.log_tabular('arrive_des_appro', average_only=True)
            test_logger.log_tabular('converge_dis', average_only=True)
            test_logger.log_tabular('out_of_range', average_only=True)
            test_logger.dump_tabular()

        # train policy
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        env.unwrapped._set_test_mode(False)
        obs, acs, rewards = [], [], []
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: np.array(o).reshape(1,-1)})
            # a = get_action_2(np.array(o))
            # save and log
            obs.append(o)
            ref_action = call_ref_controller(env, expert)
            if(epoch < pretrain_epochs):
                action = ref_action
            else:
                action = choose_action(np.array(o), True)
            
            buf.store(o, action, r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(action)
            acs.append(ref_action)
            rewards.append(r)

            ep_ret += r
            ep_len += 1
            total_env_t += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: np.array(o).reshape(1,-1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Perform dagger and partical PPO update!
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        # pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        max_step = len(np.array(rewards))
        dagger_replay_buffer.stores(obs, acs, rewards)
        for _ in range(int(local_steps_per_epoch/10)):
                batch = dagger_replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'], tfa_ph: batch['acts']}
                q_step_ops = [dagger_pi_loss, train_dagger_pi_op]
                for j in range(10):
                    outs = sess.run(q_step_ops, feed_dict)
                logger.store(LossPi = outs[0])

        c_v_loss = sess.run(v_loss, feed_dict=inputs)
        logger.store(LossV=c_v_loss, 
                     KL=0, Entropy=0, ClipFrac=0,
                     DeltaLossPi=0,
                     DeltaLossV=0, StopIter=0)

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

    # Main loop: collect experience in env and update/log each epoch
    print(colorize("begin ppo training", 'green', bold=True))
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
            test_logger.log_tabular('arrive_des_appro', average_only=True)
            test_logger.log_tabular('converge_dis', average_only=True)
            test_logger.log_tabular('out_of_range', average_only=True)
            test_logger.dump_tabular()

        # train policy
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        env.unwrapped._set_test_mode(False)
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: np.array(o).reshape(1,-1)})
            # a = get_action_2(np.array(o))
            # save and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: np.array(o).reshape(1,-1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Perform PPO update!
        update()

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

        
if __name__ == '__main__':
    rospy.init_node('pelican_attitude_controller_ppo_training', anonymous=True, log_level=rospy.WARN)
    import argparse
    # default_fpath = osp.join(osp.abspath(osp.pardir),'data/Pelican_position_controller_dagger_for_ppo/Pelican_position_controller_dagger_for_ppo_s3')
    default_fpath = osp.join(osp.abspath(osp.dirname(__file__)),'data/ppo_dagger/ppo_dagger_s0')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PelicanNavControllerEnv-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--seed', '-s', type=int, default=11)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--dagger_epochs', type=int, default=0)
    parser.add_argument('--pretrain_epochs', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=0)
    parser.add_argument('--activation', type=str, default=tf.nn.relu)
    parser.add_argument('--output_activation', type=str, default=tf.nn.tanh)
    parser.add_argument('--exp_name', type=str, default='ppo_dagger_with_turbulence')
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

    ppo(lambda : env, policy_path=default_fpath, expert=ref_controller, actor_critic=core.mlp_actor_critic_m,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, activation=args.activation, output_activation=args.output_activation), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        dagger_epochs=args.dagger_epochs, pretrain_epochs=args.pretrain_epochs, save_freq=args.save_freq, logger_kwargs=logger_kwargs)