# -*- coding: utf-8 -*-
import abc
import gym
from gym import spaces
from pygame import locals
import numpy as np
import deepmind_lab
import cv2


def _action(*entries):
    return np.array(entries, dtype=np.intc)


class DeepmindLabEnvironment(gym.Env):

    def __init__(self,
                 level,
                 configs,
                 observation_keys,
                 height=84,
                 width=84,
                 frame_skip=4,
                 fps=60,
                 ):
        """
        Create the deepmind lab environment with level
        :param level: level
        :param configs: config
        :param observation_keys: available observations
        :param height: height for RGB observation
        :param width: width for RGB observation
        :param frame_skip: frame skip
        :param fps:
        """
        super(DeepmindLabEnvironment, self).__init__()
        # 相关的属性
        self.level = level
        self.viewer = None
        self.frame_skip = frame_skip
        config = {
            'fps': str(fps),
            'width': str(width),
            'height': str(height),
        }
        config.update(configs)
        env = deepmind_lab.Lab(level,
                               observation_keys,
                               config=config)
        self.lab = env

    @abc.abstractmethod
    def reset(self, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, action):
        pass

    def close(self):
        self.lab.close()
        # print("lab environment stopped, returned ", 0)

    @abc.abstractmethod
    def render(self, mode='human'):
        pass

    @abc.abstractmethod
    def get_action_meanings(self):
        pass

    def observation_space(self):
        return self.lab.observation_spec()

    def action_space(self):
        return self.lab.action_spec()


class DeepmindLabMazeNavigationEnvironment(DeepmindLabEnvironment):

    # Define the Navigation-like deepmind lab environments
    metadata = {'render.modes': ['rgb_array', 'rgbd_array', 'human']}
    # action: keys, meanings
    # 定义的是基于导航任务的相关参数
    NAV_ACTION_MEANING = {
        0: "look_left",
        1: "look_right",
        2: "strafe_left",
        3: "strafe_right",
        4: "forward",
        5: "backward"
    }

    NAV_KEY_TO_ACTION = {
        (locals.K_4,): 0,
        (locals.K_6,): 1,
        (locals.K_LEFT,): 2,
        (locals.K_RIGHT,): 3,
        (locals.K_UP,): 4,
        (locals.K_DOWN,): 5
    }
    # gym-like action space
    ACTION_LIST = [
        _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
        _action(20, 0, 0, 0, 0, 0, 0),  # look_right
        # _action(  0,  10,  0,  0, 0, 0, 0), # look_up
        # _action(  0, -10,  0,  0, 0, 0, 0), # look_down
        _action(0, 0, -1, 0, 0, 0, 0),  # strafe_left
        _action(0, 0, 1, 0, 0, 0, 0),  # strafe_right
        _action(0, 0, 0, 1, 0, 0, 0),  # forward
        _action(0, 0, 0, -1, 0, 0, 0),  # backward
        # _action(  0,   0,  0,  0, 1, 0, 0), # fire
        # _action(  0,   0,  0,  0, 0, 1, 0), # jump
        # _action(  0,   0,  0,  0, 0, 0, 1)  # crouch
    ]
    DEFAULT_ACTION = _action(0, 0, 0, 0, 0, 0, 0)

    def __init__(self, level, width=84, height=84, frame_skip=4, fps=60, enable_depth=False, other_configs=None, other_obs=None):
        """
        Navigation task
        :param level:
        :param width:
        :param height:
        :param frame_skip:
        :param fps:
        :param enable_depth:
        :param other_configs:
        :param other_obs:
        """
        configs = dict()
        if other_configs is not None:
            configs.update(other_configs)
        # set obs
        if enable_depth:
            obs = ['RGBD_INTERLEAVED']
        else:
            obs = ['RGB_INTERLEAVED']
        self._obs_key = obs[-1]
        if other_configs is not None:
            obs.extend(other_obs)
        self._enable_depth = enable_depth
        super(DeepmindLabMazeNavigationEnvironment, self).__init__(level=level,
                                                                   configs=configs,
                                                                   observation_keys=obs,
                                                                   height=height,
                                                                   width=width,
                                                                   frame_skip=frame_skip,
                                                                   fps=fps)
        # 定义各种 space
        self.action_space = gym.spaces.Discrete(len(self.ACTION_LIST))
        if enable_depth:
            self.observation_space = spaces.Box(0, 255, shape=[height, width, 4], dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(0, 255, shape=[height, width, 3], dtype=np.uint8)
        self.last_state = None

    def get_action_meanings(self):
        return [self.NAV_ACTION_MEANING[i] for i in range(0, self.action_space.n)]

    def _prepare_for_rgb(self, img):
        if self._enable_depth:
            img = img[:, :, :3]
        return img

    def render(self, mode='human'):
        rgb_or_rgbd = self.lab.observations()[self._obs_key]
        if mode == 'rgb_array':
            return self._prepare_for_rgb(img=rgb_or_rgbd)
        elif mode == 'rgbd_array':
            assert self._enable_depth, 'please enable depth output when initialize the object'
            return rgb_or_rgbd
        elif mode == 'human':
            # pop up a window and render
            img = self._prepare_for_rgb(rgb_or_rgbd)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # TODO deepmind lab has builtin BGR observation space
            cv2.imshow('deepmind lab', img)
            cv2.waitKey(1)
        else:
            super(DeepmindLabMazeNavigationEnvironment, self).render(mode=mode)  # just raise an exception

    def step(self, action):
        """
        perform an action
        :param action:
        :return:
        """

        if action < 0 or action >= len(self.ACTION_LIST):
            real_action = self.DEFAULT_ACTION
        else:
            real_action = self.ACTION_LIST[action]
        reward = self.lab.step(real_action, num_steps=self.frame_skip)
        terminated = not self.lab.is_running()
        if terminated:
            state, info = np.copy(self.last_state), dict()  # just use the last observation
        else:
            obs = self.lab.observations()
            state = obs[self._obs_key]
            self.last_state = state
            del obs[self._obs_key]  # todo del the obs key
            info = obs
        return state, reward, terminated, info

    def reset(self, **kwargs):
        self.lab.reset(seed=None)
        obs = self.lab.observations()
        state = np.copy(obs[self._obs_key])
        self.last_state = state
        return state

    @staticmethod
    def get_keys_to_action():
        """
        for playing
        :return:
        """
        return DeepmindLabMazeNavigationEnvironment.NAV_KEY_TO_ACTION
