# Polymetis

## Usage

### Start the **Controller Manager Server** and **Robot Client**

See instructions in [examples](https://github.com/facebookresearch/polymetis/tree/master/examples/) to run the server and client. Clients include:

- **Franka Client**: A hardware client designed to work with the Franka Panda arm, implemented as a wrapper around [Libfranka](https://frankaemika.github.io/docs/libfranka.html)
- **Simulation Client**: A simulation client which has the same interface as the hardware client. For details, see documentation in [polysim](polysim-doc).
- **Empty Statistics Client**: A robot client which prints statistics about time taken by RPC calls and always returns joint states of 0s). Can be used to confirm that the Controller Manager Server is running properly and to debug performance issues.

[Hydra](http://hydra.cc/) configuration files can be found in [polymetis/conf/robot_client](https://github.com/facebookresearch/polymetis/tree/master/polymetis/conf/robot_client).

### Connect a **User Client** to the Controller Manager Server & use it to execute user scripts

The user API is exposed through `RobotInterface`, which is initialized as follows:
```python
from polymetis import RobotInterface

robot = RobotInterface(
    ip_address="localhost",
)
```
`ip_address` is the IP of the Controller Manager Server. Since we're launching it locally in this example, the IP will be `localhost`.

Definitions of available methods can be found [here](https://github.com/facebookresearch/polymetis/tree/master/polymetis/python/polymetis/robot_interface.py).
Sample user scripts can be found in [examples](https://github.com/facebookresearch/polymetis/tree/master/examples).
