import os
import torch

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register


class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    PENDULUM_LENGTH = 0.6

    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/cartpole.xml' % dir_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, CartpoleEnv.PENDULUM_LENGTH]))) / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        return ob, reward, done, {}

    def reset(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    @staticmethod
    def _get_ee_pos(x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 - CartpoleEnv.PENDULUM_LENGTH * np.sin(theta),
            -CartpoleEnv.PENDULUM_LENGTH * np.cos(theta)
        ])


class CartpoleConfigModule:

    def __init__(self, device):
        self.device = device
        self.ee_sub = torch.tensor([0.0, 0.6], device=device, dtype=torch.float)

    def obs_cost_fn(self, obs):
        ee_pos = self._get_ee_pos(obs)

        ee_pos -= self.ee_sub

        ee_pos = ee_pos ** 2

        ee_pos = - ee_pos.sum(dim=-1)

        return - (ee_pos / (0.6 ** 2)).exp()

    def ac_cost_fn(self, acs):
        return 0.01 * (acs ** 2).sum(dim=-1)

    def _get_ee_pos(self, obs):
        x0, theta = obs[..., :1], obs[..., 1:2]

        return torch.cat([
            x0 - 0.6 * theta.sin(), -0.6 * theta.cos()
        ], dim=-1)

register(
    id='MBRLCartpole-v0',
    entry_point='src.cartpole_env:CartpoleEnv'
)