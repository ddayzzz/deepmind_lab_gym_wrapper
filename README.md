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

## Enable some useful outputs:

```python
import gym
import deepmind_lab_as_gym

env = gym.make('DeepmindLabSeekavoidArena01-v0', enable_velocity=True)  # output velocity

# Use the environment
observation = env.reset()  # Note that observation is a gym's Dict instance that contains both RGB and velocity outputs
```

## Optional arguments

Different arguments for extra observation enables different observation outputs which is mapped by the corresponding key name 


Optional Arguments|Effects|Extra observation key name(s)
-|-|-
frame_skip|Please refer to Deepmind lab Python API|-
seed|Please refer to Deepmind lab Python API|-
enable_velocity|Output agent's current velocity|**velocity**
enable_top_down_view|Output debug top down view|**top_down** and **word_position**
enable_depth|Depth channel|**depth**
top_down_width|[Debug observation](https://github.com/deepmind/lab/blob/master/docs/users/observations.md#debug-observations-player-only)|-
top_down_height|[Debug observation](https://github.com/deepmind/lab/blob/master/docs/users/observations.md#debug-observations-player-only)|-

## Action and observation space:

```python
env = gym.make('DeepmindLabNavMazeStatic01-v0', enable_velocity=True, enable_top_down_view=True, enable_depth=True)
```
Action space is `Discrete(6)`

Observation space is `Dict(depth:Box(84, 84), rgb:Box(84, 84, 3), top_down:Box(160, 160, 3), velocity:Box(1, 6), word_position:Box(1, 3))`

Key name|observation
-|-
velocity|`Box(1,6), float`. I concatenate `VEL.TRANS` and `VEL.ROT` from the inner observation output.
top_down|`Box(top_down_width, top_down_height, 3)`, `uint8`. It comes from a debug observation named `DEBUG.CAMERA.TOP_DOWN`
word_position|`Box(1, 3), float` from `DEBUG.POS.TRANS`
depth|`Box(84,84), uint8`. I slice the inner observation `RGBD_INTERLEAVED` at last axis
rgb|`Box(84,84,3), uint8`. It the basic observation that can be represent as curret state of the player/agent.

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
```

# Thanks

- [jkulhanek/gym-deepmindlab-env](https://github.com/jkulhanek/gym-deepmindlab-env)
- [Deepmind lab: Python API](https://github.com/deepmind/lab/blob/master/docs/users/python_api.md)
- [Deepmind lab: Debug observation](https://github.com/deepmind/lab/blob/master/docs/users/observations.md#debug-observations-player-only)
- [Deepmind lab: Custom observations](https://github.com/deepmind/lab/blob/master/docs/users/observations.md#custom-observations-player-only)