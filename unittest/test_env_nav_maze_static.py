# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from itertools import count

import gym
import deepmind_lab_as_gym


class TestNavMaze(unittest.TestCase):

    def _init_env(self, env_name, **kwargs):
        env = gym.make(id=env_name)
        return env

    def test_process(self):
        env_names = ['DeepmindLabNavMazeStatic01-v0', 'DeepmindLabNavMazeStatic03-v0']
        max_episodes = 1
        for i, env_name in enumerate(env_names):
            print('Test: ', env_name)
            env = self._init_env(env_name=env_name)
            episodes = 0
            try:
                while episodes < max_episodes:
                    env.reset()
                    episodic_reward = 0.0
                    time = 0
                    for t in count():
                        env.render()
                        action = np.random.choice(env.action_space.n)

                        obs, reward, terminal, _ = env.step(action)
                        episodic_reward += reward
                        if terminal:
                            time = t
                            break
                    print('time: {}, episodic reward: {}'.format(time, episodic_reward))
                    episodes += 1
            except KeyboardInterrupt:
                pass
            finally:
                env.close()
            self.assertTrue(episodes == max_episodes)

if __name__ == '__main__':
    unittest.main()