# -*- coding: utf-8 -*-
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import deepmind_lab
import cv2


def _action(*entries):
    return np.array(entries, dtype=np.intc)


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


class DeepmindLabEnvironment(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'rgbd_array', 'human']}

    def __init__(self,
                 level,
                 height=84,
                 width=84,
                 frame_skip=4,
                 channel_first=False,
                 enable_velocity=False,
                 enable_top_down_view=False,
                 top_down_width=160,
                 top_down_height=160,
                 enable_depth=False,
                 fps=60,
                 **kwargs):
        """
        Create a deepmind lab environment
        :param level: level for deepmind lab
        :param frame_skip:
        :param channel_first: this will change the observation_space.
        :param enable_velocity: velocity
        :param enable_top_down_view:
        :param top_down_width:
        :param top_down_height:
        :param kwargs: the optional config for deepmind lab
        """
        super(DeepmindLabEnvironment, self).__init__()
        # 相关的属性
        self.level = level
        self.viewer = None
        self.enable_top_down_view = enable_top_down_view
        self.enable_velocity = enable_velocity
        self.enable_depth = enable_depth
        self.frame_skip = frame_skip
        self.channel_first = channel_first
        # check arguments is current?
        # config the observation_space
        basic_obs = []
        if enable_depth:
            if channel_first:
                basic_obs.append('RGBD')
            else:
                basic_obs.append('RGBD_INTERLEAVED')
        else:
            if channel_first:
                basic_obs.append('RGB')
            else:
                basic_obs.append('RGB_INTERLEAVED')
        config = {
            'fps': str(fps),
            'width': str(width),
            'height': str(height),
        }
        self._obs_key = basic_obs[0]
        if enable_top_down_view:
            basic_obs.extend(['DEBUG.CAMERA.TOP_DOWN', 'DEBUG.POS.TRANS', ])
            config.update(maxAltCameraWidth=str(top_down_width), maxAltCameraHeight=str(top_down_height),
                          hasAltCameras='true')
        if enable_velocity:
            basic_obs.extend(['VEL.TRANS', 'VEL.ROT'])  # 速度
        # 添加其他参数给 config
        config.update(**kwargs)
        env = deepmind_lab.Lab(level,
                               basic_obs,
                               config=config)
        self.lab = env
        # action_space
        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        # observation_space
        if enable_depth:
            if channel_first:
                self.observation_space = spaces.Box(0, 255, shape=[4, height, width], dtype=np.uint8)
            else:
                self.observation_space = spaces.Box(0, 255, shape=[height, width, 4], dtype=np.uint8)
        else:
            if channel_first:
                self.observation_space = spaces.Box(0, 255, shape=[3, height, width], dtype=np.uint8)
            else:
                self.observation_space = spaces.Box(0, 255, shape=[height, width, 3], dtype=np.uint8)
        #
        self.last_observation = None
        # seed
        self.seed()

    def reset(self, **kwargs):
        self.lab.reset(seed=self.np_random.choice(2147483647))  # max of int32
        obs = self.lab.observations()
        self.last_observation, _ = self._get_state(obs)
        return self.last_observation

    def _get_state(self, obs):
        returned = dict()
        # if self.enable_depth:
        #     rgb_state = obs['RGBD_INTERLEAVED'][:, :, :3]
        #     returned['depth'] = obs['RGBD_INTERLEAVED'][:, :, 3]
        # else:
        #     rgb_state = obs['RGB_INTERLEAVED']
        rgb_or_rgbd = obs[self._obs_key]
        if self.enable_top_down_view:
            returned['top_down'] = obs['DEBUG.CAMERA.TOP_DOWN']
            returned['word_position'] = obs['DEBUG.POS.TRANS']

        if self.enable_velocity:
            returned['velocity'] = np.concatenate([obs['VEL.TRANS'], obs['VEL.ROT']])
        return rgb_or_rgbd, returned

    def step(self, action):
        """
        进行一个动作
        :param action:
        :return: state,
        """

        if action < 0 or action >= len(ACTION_LIST):
            real_action = DEFAULT_ACTION
        else:
            real_action = ACTION_LIST[action]

        # self.connection.send([LabWorkCommand.ACTION, real_action])
        # obs, reward, terminated = self.connection.recv()
        reward = self.lab.step(real_action, num_steps=self.frame_skip)
        terminated = not self.lab.is_running()
        if terminated:
            state, info = np.copy(self.last_observation), dict()  # just use the last observation
        else:
            obs = self.lab.observations()
            state, info = self._get_state(obs=obs)
            self.last_observation = state
        return state, reward, terminated, info

    def close(self):
        self.lab.close()
        # print("lab environment stopped, returned ", 0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _prepare_for_rgb(self, img):
        if self.channel_first:
            if self.enable_depth:
                img = img[:3, :, :]  # delete depth
            img = np.transpose(img, axes=[1, 2, 0])  # to RGB
        else:
            if self.enable_depth:
                img = img[:, :, :3]
        return img

    def render(self, mode='human'):
        rgb_or_rgbd = self.lab.observations()[self._obs_key]
        if mode == 'rgb_array':
            return self._prepare_for_rgb(img=rgb_or_rgbd)
        elif mode == 'rgbd_array':
            assert self.enable_depth, 'please enable depth output when initialize the object'
            if self.channel_first:
                rgb_or_rgbd = np.transpose(rgb_or_rgbd, axes=[1, 2, 0])  # channel first to RGBD
            return rgb_or_rgbd
        elif mode == 'human':
            # pop up a window and render
            img = self._prepare_for_rgb(rgb_or_rgbd)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # TODO deepmind lab has builtin BGR observation space
            cv2.imshow('deepmind lab', img)
            cv2.waitKey(1)
        else:
            super(DeepmindLabEnvironment, self).render(mode=mode)  # just raise an exception

    def get_action_meanings(self):
        return [
            'look_left',
            'look_right',
            'strafe_left',
            'strafe_right',
            'forward',
            'backward'
        ]

