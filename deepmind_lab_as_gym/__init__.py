import os
from gym import register, make


LEVELS = ['lt_chasm', 'lt_hallway_slope', 'lt_horseshoe_color', 'lt_space_bounce_hard', \
              'nav_maze_random_goal_01', 'nav_maze_random_goal_02', 'nav_maze_random_goal_03', 'nav_maze_static_01', \
              'nav_maze_static_02', 'nav_maze_static_03', 'seekavoid_arena_01', 'stairway_to_melon']


def _find_levels():
    # TODO add more levels
    return LEVELS


def _to_pascal(text):
    return ''.join(map(lambda x: x.capitalize(), text.split('_')))


MAP = {_to_pascal(l): l for l in _find_levels()}

for key, l in MAP.items():
    register(
        id='DeepmindLab%s-v0' % key,
        entry_point='deepmind_lab_as_gym.env:DeepmindLabEnvironment',
        kwargs=dict(level=l)
    )
# register a deepmind standard environment, users need pass the proper arguments when use it
register(id='DeepmindLab-v0', entry_point='deepmind_lab_as_gym.env:DeepmindLabEnvironment')