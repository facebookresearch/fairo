# Example 08: GUI

In this example we show off multiple gui techniques, as well as environmental variables.

`conda_proc` simply shows off the `env` argument as a way of passing along environmental variables, regardless of runtime.

`docker_x11_proc` shows off docker runtime flags, mounting and environmental flags. Environmental passed through `env` and through the passthrough kwargs are properly merged.

`docker_opengl_proc` shows how to use the nvidia runtime.
