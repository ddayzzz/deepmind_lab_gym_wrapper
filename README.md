# Gym wrapper for DeepMind Lab environments

This is a gym-like wrapper for Deepmind lab environment. This repo is inspired by `jkulhanek/gym-deepmindlab-env` and I add some observation options which might helpful!

# Usage

According to Deepmind lab's documents, we defined some task-specific wrappers:

environment id|Class names
-|-
DeepmindLabEnv-v0|DeepmindLabEnvironment
DeepmindLabNavEnv-v0|DeepmindLabMazeNavigationEnvironment

Please note that `DeepmindLabMazeNavigationEnvironment` is a class extends `DeepmindLabEnvironment` and implements abstract methods in base class.

Gym-like abstract methods:

- `reset`
- `step`
- `render`
- `get_action_meanings`


## Common use:

```python
import gym
import deepmind_lab_as_gym

env = gym.make(id='DeepmindLabNavEnv-v0', level=level)

# Use the environment
observation = env.reset()
```

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