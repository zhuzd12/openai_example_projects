import numpy as np
from gym import utils
from openai_ros.task_envs.pelican import pelican_attitude_controller

class PelicanControlEnvNew(pelican_attitude_controller.PelicanAttControllerEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)

    def _step_new(self, action):
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs_new(self):
        return np.array(self.get_full_obs())

    def reset_model(self):
        return self.reset()

    def viewer_setup(self):
        pass