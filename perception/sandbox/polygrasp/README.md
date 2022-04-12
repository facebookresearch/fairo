# Polygrasp

## Installation

```bash
pip install ../../realsense_driver
pip install -e .
```

## Development

### Recompiling grasping gRPC server

```bash
# from root directory
python -m grpc_tools.protoc -I src --python_out=src --grpc_python_out=src polygrasp/polygrasp.proto
```

Note that it's important to point the arguments of `-I`, `--python_out`, `grpc_python_out` to `.` so that the `.proto` argument uses the full directory structure instead of attempting to import the `_pb2` file without relative imports, as described in [this comment](https://github.com/grpc/grpc/issues/9575#issuecomment-293934506).
