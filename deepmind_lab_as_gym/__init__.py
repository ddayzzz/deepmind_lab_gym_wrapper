import os
from gym import register, make

# register a deepmind standard environment, users need pass the proper arguments when use it
register(id='DeepmindLabNavEnv-v0', entry_point='deepmind_lab_as_gym.env:DeepmindLabMazeNavigationEnvironment')