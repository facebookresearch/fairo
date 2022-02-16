# Troubleshooting

## Cannot connect to Franka Robot

https://frankaemika.github.io/docs/troubleshooting.html#running-a-libfranka-executable-fails-with-connection-timeout


## Motion stopped due to discontinuities or `communication_constraints_violation`

### Networking

https://frankaemika.github.io/docs/troubleshooting.html#running-a-libfranka-executable-fails-with-connection-timeout

### CPU performance

Disabling CPU frequency scaling: https://frankaemika.github.io/docs/troubleshooting.html#disabling-cpu-frequency-scaling

### End-to-end testing

[`scripts/benchmark_pd_control.py`](https://github.com/facebookresearch/fairo/tree/main/polymetis/scripts/benchmark_pd_control.py)


## Inaccurate tracking performance

### End-effector configuration
Update Desk and Polymetis URDF with the correct end-effector

### Controller parameters
Tune gains