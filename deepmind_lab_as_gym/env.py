# -*- coding: utf-8 -*-
import gym
from gym import spaces
import numpy as np
import deepmind_lab


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
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self,
                 level,
                 frame_skip=4,
                 seed=None,
                 enable_velocity=False,
                 enable_top_down_view=False,
                 top_down_width=160,
                 top_down_height=160,
                 enable_depth=False,
                 **kwargs):
        """

        :param level: level for deepmind lab
        :param frame_skip:
        :param seed:
        :param enable_velocity:
        :param enable_top_down_view:
        :param top_down_width:
        :param top_down_height:
        :param kwargs:
        """
        super(DeepmindLabEnvironment, self).__init__()
        # 相关的属性
        self.fixed_seed = seed
        self.level = level
        self.viewer = None
        self.enable_top_down_view = enable_top_down_view
        self.enable_vel = enable_velocity
        self.frame_skip = frame_skip
        self.enable_depth = enable_depth
        # check arguments is current?
        # config the observation_space
        basic_obs = [
            'RGB_INTERLEAVED'
        ]

        config = {
            'fps': str(60),
            'width': str(84),
            'height': str(84),
        }
        if enable_top_down_view:
            basic_obs.extend(['DEBUG.CAMERA.TOP_DOWN', 'DEBUG.POS.TRANS', ])
            config.update(maxAltCameraWidth=str(top_down_width), maxAltCameraHeight=str(top_down_height),
                          hasAltCameras='true')
        if enable_depth:
            basic_obs.append('RGBD_INTERLEAVED')
            # del basic_obs[0]  # todo i can just use the RGBD for RGB output
        if enable_velocity:
            basic_obs.extend(['VEL.TRANS', 'VEL.ROT'])  # 速度

        env = deepmind_lab.Lab(level,
                               basic_obs,
                               config=config)
        self.lab = env
        # action_space
        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        # observation_space
        obs_spaces = {'rgb': spaces.Box(0, 255, shape=[84, 84, 3], dtype=np.uint8)}

        if enable_top_down_view:
            # top_down = [w, h, 3]
            # word_pos = [x, y, z]: double
            obs_spaces['top_down'] = spaces.Box(0, 255, shape=[top_down_width, top_down_height, 3], dtype=np.uint8)
            obs_spaces['word_position'] = spaces.Box(low=0.0, high=np.finfo(np.float).max, shape=[1, 3], dtype=np.float)
        if enable_depth:
            obs_spaces['depth'] = spaces.Box(0, 255, shape=[84, 84], dtype=np.uint8)
        if enable_velocity:
            obs_spaces['velocity'] = spaces.Box(0.0, high=np.finfo(np.float).max, shape=[1, 6], dtype=np.float)

        self.observation_space = spaces.Dict(spaces=obs_spaces)

    def reset(self, **kwargs):
        self.lab.reset(seed=self.fixed_seed)
        obs = self.lab.observations()
        state = self._get_state(obs)
        return state

    def _get_state(self, obs):
        returned = dict(rgb=obs['RGB_INTERLEAVED'])

        if self.enable_top_down_view:
            returned['top_down'] = obs['DEBUG.CAMERA.TOP_DOWN']
            returned['word_position'] = obs['DEBUG.POS.TRANS']

        if self.enable_depth:
            returned['depth'] = obs['RGBD_INTERLEAVED'][:, :, 3]

        if self.enable_vel:
            returned['velocity'] = np.concatenate([obs['VEL.TRANS'], obs['VEL.ROT']])
        return returned

    def step(self, action):
        """
        进行一个动作
        :param action:
        :return: state,
        """

        if action < 0:
            real_action = DEFAULT_ACTION
        else:
            real_action = ACTION_LIST[action]

        # self.connection.send([LabWorkCommand.ACTION, real_action])
        # obs, reward, terminated = self.connection.recv()
        reward = self.lab.step(real_action, num_steps=self.frame_skip)
        terminated = not self.lab.is_running()
        if not terminated:
            obs = self.lab.observations()
        else:
            obs = 0
        # step 返回的信息
        if terminated:
            state = None
        else:
            state = self._get_state(obs=obs)

        return state, reward, terminated, None

    def close(self):
        self.lab.close()
        # print("lab environment stopped, returned ", 0)

    def render(self, mode='human'):
        rgb = self.lab.observations()['RGB_INTERLEAVED']
        if mode == 'rgb_array':
            return rgb
        elif mode is 'human':
            # pop up a window and render
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(rgb)
            return self.viewer.isopen
        else:
            super(DeepmindLabEnvironment, self).render(mode=mode)  # just raise an exception

