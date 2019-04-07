#!/usr/bin/env python3

import gym
import tensorflow as tf
import time
from distutils.dir_util import copy_tree
import os
import os.path as osp
import json
import yaml
# import actor_critic_core

import random
import numpy as np
import memory
from openai_ros.task_envs.pelican import pelican_attitude_controller
from openai_ros.task_envs.pelican import pelican_randomquat_position_controller
from openai_ros.task_envs.pelican import pelican_attitude_controller_test

# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from planning_msgs.srv import AttitudeControllerService
import pyquaternion
from spinup.utils.run_utils import setup_logger_kwargs
import spinup.algos.ppo.core as core
from spinup.utils.logx import EpochLogger

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.ref_act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        # self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # self.val_buf = np.zeros(size, dtype=np.float32)
        # self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, ref_act, rew):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        # self.val_buf[self.ptr] = val
        # self.logp_buf[self.ptr] = logp
        self.ref_act_buf[self.ptr] = ref_act
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
        # vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        # deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def sample_batch(self, batch_size=64):
        idxs = np.random.randint(0, self.max_size, size=batch_size)
        return [self.obs_buf[idxs], self.act_buf[idxs], self.ref_act_buf[idxs],
                self.ret_buf[idxs]]

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.ref_act_buf,
                self.ret_buf]

def call_mpc(env, mpc_service):
    odo = env.unwrapped.get_odom()
    target_pose = env.unwrapped.get_target_pose()
    try:
        res = mpc_service(odo, target_pose)
    except (rospy.ServiceException) as e:
        print("/pelican/call_nlmpc_controller service call failed")
    return [res.cmd.yaw_rate, res.cmd.roll, res.cmd.pitch, res.cmd.thrust.z]

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        os.unlink(file)

if __name__ == '__main__':
    rospy.init_node('pelican_ppo_attitude_controller_dagger_training', anonymous=True, log_level=rospy.WARN)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PelicanAttControllerEnv-v0')
    parser.add_argument('--test_env', type=str, default='PelicanAttControllerTestEnv-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--local_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--train_pi_iters', type=int, default=20)
    parser.add_argument('--train_v_iters', type=int, default=20)
    parser.add_argument('--test_frequency', type=int, default=10)
    parser.add_argument('--test_epochs', type=int, default=91)
    parser.add_argument('--add_noise', type=bool, default=True)
    parser.add_argument('--activation', type=str, default=tf.tanh)
    parser.add_argument('--output_activation', type=str, default=tf.tanh)
    parser.add_argument('--yaml_file', type=str, default="dagger")
    parser.add_argument('--exp_name', type=str, default='Pelican_position_controller_dagger_for_ppo')
    args = parser.parse_args()

    # env = gym.make('PelicanAttControllerEnv-v0')
    env = gym.make(args.env)
    A_DIM = env.unwrapped.a_dim
    S_DIM = env.unwrapped.s_dim
    # if args.test_env is not None:
    #     test_env = gym.make(args.test_env)
    #     assert(A_DIM == test_env.unwrapped.a_dim)
    #     assert(S_DIM == test_env.unwrapped.s_dim)
    # else:
    #     test_env = env
    # test_env = env
    
    rospy.loginfo("Gym environment done")
    outdir = '/tmp/openai_ros_experiments/'
    # env = gym.wrappers.Monitor(env, outdir, resume=True)
    # test_env = gym.wrappers.Monitor(test_env, outdir, resume=True)
    call_nlmpc = rospy.ServiceProxy('/pelican/call_nlmpc_controller', AttitudeControllerService)

    yaml_path = os.path.abspath('yaml_files/'+args.yaml_file+'.yaml')
    assert(os.path.exists(yaml_path))
    with open(yaml_path, 'r') as f:
        params = yaml.load(f)
    
    # set train epoch parameters
    

    epochs = args.epochs
    pre_train_epochs = args.pretrain_epochs
    trainning_policy_epochs = args.test_frequency
    test_policy_epochs = args.test_epochs
    episode_steps = 500

    # load polocy network parameters
    # I_learningRate = params["learning_rate"]
    pi_lr = params["actor_learning_rate"]
    vf_lr = params["critic_learning_rate"]
    minibatch_size = params["batch_size"]
    discountFactor = params["discount_factor"]
    explorationRate = params["exploration_rate"]
    memorySize = params["memory_size"]
    hidden_sizes = params['hidden_sizes']
    act_noise_amount = params["action_noise"]
    policy_net_training_steps = params["training_steps"]
    # print("hidden_sizes: ", hidden_sizes)


    DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir = DEFAULT_DATA_DIR)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(args)
    # policy_net = actor_critic_core.Policy_Net(env, EP_MAX=epochs, EP_LEN=episode_steps, GAMMA=discountFactor, AR=I_learningRate, CR=I_learningRate, BATCH=minibatch_size, UPDATE_STEP=policy_net_training_steps, 
    #     hidden_sizes=hidden_sizes, activation=args.activation, output_activation=args.output_activation, act_noise_amount=act_noise_amount, logger_kwargs=logger_kwargs)

    ###### set network
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    t_a_ph = core.placeholder_from_space(env.action_space)
    ret_ph = core.placeholder(None)

    mu, pi, logp, logp_pi, v = core.mlp_actor_critic_m(x_ph, a_ph, hidden_sizes=hidden_sizes, activation=args.activation, output_activation=args.output_activation, action_space=env.action_space)
    all_phs = [x_ph, a_ph, t_a_ph, ret_ph]

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # dagger objectives
    pi_loss = tf.reduce_mean(tf.square(mu -  t_a_ph))
    v_loss = tf.reduce_mean((ret_ph - v)**2)

     # Optimizers
    train_pi =  tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v =  tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

     # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'mu': mu, 'pi': pi, 'v': v})

    # replay_buffer = ReplayBuffer(S_DIM, A_DIM, memorySize)
    local_steps_per_epoch = args.local_steps_per_epoch
    replay_buffer = PPOBuffer(S_DIM, A_DIM, local_steps_per_epoch, args.gamma, args.lam)

    train_pi_iters=args.train_pi_iters
    train_v_iters=args.train_v_iters
    def update():
        # inputs = {k:v for k,v in zip(all_phs, replay_buffer.sample_batch(minibatch_size))}
        test_inputs = {k:v for k,v in zip(all_phs, replay_buffer.sample_batch(minibatch_size))}
        pi_l_old, v_l_old = sess.run([pi_loss, v_loss], feed_dict=test_inputs)

        # Training
        for i in range(train_pi_iters):
            inputs = {k:v for k,v in zip(all_phs, replay_buffer.sample_batch(minibatch_size))}
            [sess.run(train_pi, feed_dict=inputs) for j in range(policy_net_training_steps)]
        for _ in range(train_v_iters):
            inputs = {k:v for k,v in zip(all_phs, replay_buffer.sample_batch(minibatch_size))}
            [sess.run(train_v, feed_dict=inputs) for j in range(policy_net_training_steps)]
        replay_buffer.get()

        # Log changes from update
        pi_l_new, v_l_new = sess.run([pi_loss, v_loss], feed_dict=test_inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()

    # start iterating from 'current epoch'.
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    for epoch in range(epochs):
        env.unwrapped._set_test_mode(False)
        # observation = env.reset()
        # observation, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        # number of timesteps
        for t in range(local_steps_per_epoch):
            # action = env.action_space.sample()
            ref_action = call_mpc(env, call_nlmpc)
            if(epoch < pre_train_epochs):
                action = ref_action
            else:
                # action = policy_net.choose_action(np.array(observation), True)
                action = sess.run(mu, feed_dict={x_ph: np.array(o).reshape(1,-1)})[0]
                if args.add_noise:
                    noise = act_noise_amount * env.action_space.high * np.random.normal(size=action.shape)
                    action = action + noise

            replay_buffer.store(o, action, ref_action, r)

            o, r, d, info = env.step(action)
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == episode_steps)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: np.array(o).reshape(1,-1)})
                replay_buffer.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            
        # Save model
        if (epoch % trainning_policy_epochs == 0) or (epoch == epochs-1):
            logger.save_state({}, None)

        # Perform PPO update!
        update()

        # test policy
        if epoch % trainning_policy_epochs == 0 and epoch > 0:
            env.unwrapped._set_test_mode(True)
            for i in range(test_policy_epochs):
                observation = env.reset()
                policy_cumulated_reward = 0
                p_buffer_s, p_buffer_a = [], []
                for t in range(episode_steps):
                    sess.run(mu, feed_dict={x_ph: np.array(observation).reshape(1,-1)})
                    
                    newObservation, reward, done, info = env.step(action)
                    observation = newObservation
                    if (t == episode_steps-1):
                        print ("reached the end")
                        done = True
                    # env._flush(force=True)
                    policy_cumulated_reward += reward

                    if done:
                        m, s = divmod(int(time.time() - start_time), 60)
                        h, m = divmod(m, 60)
                        # print("EP " + str(i) + " - {} test number".format(epoch // trainning_policy_epochs) + " - CReward: " + str(round(policy_cumulated_reward, 4)) + "  Time: %d:%02d:%02d" % (h, m, s))
                        logger.store(policy_reward=policy_cumulated_reward)
                        logger.store(policy_steps=t)
                        break
                    else:
                        logger.store(action_error=info['action_error'])
            
            logger.log_tabular('epoch', epoch)
            logger.log_tabular('policy_reward', average_only=True)
            logger.log_tabular('policy_steps', average_only=True)
            logger.log_tabular('action_error', with_min_and_max=False)
            logger.log_tabular('LossPi', average_only=True)
            logger.dump_tabular()
            # save model
            logger.save_state({}, None)

    env.close()