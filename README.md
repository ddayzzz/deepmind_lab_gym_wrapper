# Gym wrapper for DeepMind Lab environments

This is a gym-like wrapper for Deepmind lab environment. This repo is inspired by `jkulhanek/gym-deepmindlab-env` and I add some observation options which might helpful!

# Usage

## Common use:
```python
import gym
import deepmind_lab_as_gym

env = gym.make('DeepmindLabSeekavoidArena01-v0')

# Use the environment
observation = env.reset()
```

## Optional arguments

The following table shows that three observations are available in dict object named `info` field when call `step(action)`

Optional Arguments|Effects|Extra observation key name(s)
-|-|-
enable_velocity|Output agent's current velocity|**velocity**
enable_top_down_view|Output debug top down view|**top_down** and **word_position**

The following parameters combination will change the `observation_space`:

`enable_depth`|`channel_first`|Effect to `observation_space`
-|-|-
True|False|`Box(height,width,4), uint8`
True|True|`Box(4,height,width)`
False|False|`Box(height,width,3), uint8`
False|True|`Box(3,height,width), uint8`

You can also specify other arguments when call `gym.make` and their usage can be found in Deepmind lab documents.

## Action and observation space:

```python
env = gym.make('DeepmindLabNavMazeStatic01-v0', enable_velocity=True, enable_top_down_view=True, enable_depth=True)
```
Action space is `Discrete(6)`

Observation space depens on the combination of parameters `channel_first` and `enable_depth`

Key name in `info`|observation
-|-
velocity|`Box(1,6), float`. I concatenate `VEL.TRANS` and `VEL.ROT` from the inner observation output.
top_down|`Box(top_down_height, top_down_width, 3)`, `uint8`. It comes from a debug observation named `DEBUG.CAMERA.TOP_DOWN`
word_position|`Box(1, 3), float` from `DEBUG.POS.TRANS`


## Support environments

- DeepmindLabLtChasm-v0
- DeepmindLabLtHallwaySlope-v0
- DeepmindLabLtHorseshoeColor-v0
- DeepmindLabLtSpaceBounceHard-v0
- DeepmindLabNavMazeRandomGoal01-v0
- DeepmindLabNavMazeRandomGoal02-v0
- DeepmindLabNavMazeRandomGoal03-v0
- DeepmindLabNavMazeStatic01-v0
- DeepmindLabNavMazeStatic02-v0
- DeepmindLabNavMazeStatic03-v0
- DeepmindLabSeekavoidArena01-v0
- DeepmindLabStairwayToMelon-v0

# Installation

```shell script
python setup.py develop  # for development
python setup.py install  # for users
```

# Thanks

- [jkulhanek/gym-deepmindlab-env](https://github.com/jkulhanek/gym-deepmindlab-env)

# References

- [Deepmind lab: Python API](https://github.com/deepmind/lab/blob/master/docs/users/python_api.md)
- [Deepmind lab: Debug observation](https://github.com/deepmind/lab/blob/master/docs/users/observations.md#debug-observations-player-only)
- [Deepmind lab: Custom observations](https://github.com/deepmind/lab/blob/master/docs/users/observations.md#custom-observations-player-only)