import os
from gym import register, make

# register a deepmind standard environment, users need pass the proper arguments when use it
## default env
register(id='DeepmindLabEnv-v0', entry_point='deepmind_lab_as_gym.env:DeepmindLabEnvironment')
## navigation env
register(id='DeepmindLabNavEnv-v0', entry_point='deepmind_lab_as_gym.env:DeepmindLabMazeNavigationEnvironment')