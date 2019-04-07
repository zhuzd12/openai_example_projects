#!/usr/bin/env python3

import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from utils.logx import EpochLogger
from utils.logx import restore_tf_graph
from openai_ros.task_envs.pelican import pelican_attitude_controller
from utils.run_utils import setup_logger_kwargs
import argparse
import gym
import rospy
import rospkg
import numpy as np

def load_policy(fpath, itr='last', deterministic=False):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr
    print("itr: ", itr)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)

    return get_action

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, logger_kwargs=dict()):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger(**logger_kwargs)
    # self.logger.save_config(locals())
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        a = get_action(np.array(o))
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            print(o[9:12])
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    rospy.init_node('pelican_attitude_controller_policy_test', anonymous=True, log_level=rospy.WARN)
    parser = argparse.ArgumentParser()
    default_fpath = osp.join(osp.abspath(osp.dirname(__file__)),'data/Pelican_position_controller_dagger_for_ppo/Pelican_position_controller_dagger_for_ppo_s3')
    parser.add_argument('--exp', type=str, default="PelicanAttControllerEnv-v0")
    parser.add_argument('--fpath', type=str, default=default_fpath)
    parser.add_argument('--len', '-l', type=int, default=500)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env = gym.make(args.exp)
    outdir = '/tmp/openai_ros_experiments/'
    env = gym.wrappers.Monitor(env, outdir, force=True)
    get_action = load_policy(args.fpath, 
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic)
    logger_kwargs = setup_logger_kwargs(args.exp+'_test', args.seed)
    run_policy(env, get_action, args.len, args.episodes, logger_kwargs=logger_kwargs)