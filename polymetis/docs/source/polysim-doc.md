# Polysim

The package containing the Polymetis simulation client. Includes a wrapper around PyBullet.

## Developing custom environments

All simulation environments must inherit from the abstract simulation environment class `AbstractControlledEnv` to work with the Simulation Client.

```python
from polysim.envs import AbstractControlledEnv

class MyCustomEnv(AbstractControlledEnv):
    ... # define methods here
```

The created simulation environment must define the following functions (see [abstract_env.py](https://github.com/facebookresearch/polymetis/tree/master/polysim/polysim/envs/abstract_env.py) for details):
- `reset`
- `get_num_dofs`
- `get_current_joint_pos_vel`
- `get_current_joint_torques`
- `apply_joint_torques`

Sample simulation environments can also be found in [polysim/envs](https://github.com/facebookresearch/polymetis/tree/master/polysim/polysim/envs)

You must also define metadata around your simulation environment which instantiates a [RobotClientMetadata](https://github.com/facebookresearch/polymetis/tree/master/polymetis/python/polymetis/robot_client/metadata.py) object. An example configuration is in the [default config](https://github.com/facebookresearch/polymetis/tree/master/polymetis/conf/robot_client/franka_hardware.yaml) under `metadata_cfg`.

The Simulation Client is implemented as a wrapper around the simulation environment.
To connect a simulation environment to a local server and run for 1000 steps:
```python
from polysim import GrpcSimulationClient
from polymetis.robot_client_metadata import RobotClientMetadata

env = MyCustomEnv(...)
metadata_cfg = {"key": "value"}

sim = GrpcSimulationClient(
    env=env,
    metadata=metadata_cfg,
    ip="localhost",
)
sim.run(time_horizon=1000)
```
Notes:
- `ip` is the IP of the Controller Manager Server.
- If the `time_horizon` argument is not specified, the simulation will run indefinitely unless terminated.
