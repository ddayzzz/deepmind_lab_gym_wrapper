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

    def _init_env(self, level):
        env = gym.make(id='DeepmindLabNavEnv-v0', level=level)
        return env

    def test_process(self):
        env_names = ['nav_maze_static_01', 'nav_maze_random_goal_01']
        max_episodes = 1
        for i, env_name in enumerate(env_names):
            print('Test: ', env_name)
            env = self._init_env(level=env_name)
            episodes = 0
            try:
                while episodes < max_episodes:
                    last_frame = env.reset()
                    episodic_reward = 0.0
                    time = 0
                    for t in count():
                        env.render()
                        action = np.random.choice(env.action_space.n)

                        obs, reward, terminal, info = env.step(action)
                        self.assertTrue(id(last_frame) != id(obs))
                        last_frame = obs
                        episodic_reward += reward
                        if terminal:
                            time = t
                            self.assertTrue(obs is None)
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