# Dockerfiles

This directory contains the Dockerfile which build [`fairrobotics/kuka-workspace`](https://hub.docker.com/repository/docker/fairrobotics/kuka-workspace) Docker images, used for CircleCI.

By following the build instructions inside the Dockerfiles, you can see the necessary system-level packages and libtorch required to build `kuka-workspace`.

## Building and uploading

### Ubuntu 16

```
docker build ./ubuntu16 -t fairrobotics/kuka-workspace:ubuntu-16
docker push fairrobotics/kuka-workspace:ubuntu-16

```

### Ubuntu 18

```
docker build ./ubuntu18 -t fairrobotics/kuka-workspace:ubuntu-18
docker push fairrobotics/kuka-workspace:ubuntu-18
```
