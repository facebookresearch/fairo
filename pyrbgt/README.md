# PyRobot Vision Module

Python 3.6+ only.

## Model-based Approach: A Sparse Gaussian Approach to Region-Based 6DoF Object Tracking
Manuel Stoiber, Martin Pfanne, Klaus H. Strobl, Rudolph Triebel, and Alin Albu-Schäffer  
Best Paper Award, ACCV 2020: [paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Stoiber_A_Sparse_Gaussian_Approach_to_Region-Based_6DoF_Object_Tracking_ACCV_2020_paper.pdf), [supplementary](https://openaccess.thecvf.com/content/ACCV2020/supplemental/Stoiber_A_Sparse_Gaussian_ACCV_2020_supplemental.zip)

### Installation
**Step 1: install core dependencies**
```
sudo apt install \
    libeigen3-dev \
    libglew-dev \
    libglfw3-dev \
    libopencv-dev \
    python3-dev \
    python3-pip
```

You'll need pytorch. If you don't already have it, you can do:
```
python3 -m pip install \
  torch==1.9.0+cu111 \
  torchvision==0.10.0+cu111 \
  torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

Similar for pytorch3d:
```
python3 -m pip install \
  pytorch3d==0.5.0 \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu111_pyt190/download.html
```

**Step 2: install the package**
```
cd fairo/pyrbgt
python3 -m pip install -r ./requirements.txt
python3 -m pip install .
```

**Alternatively, you may use the provided docker image**
```
cd fairo/pyrbgt
docker build -t pyrbgt .
```

### Usage
A minimal example is provided in `examples/multi_obj_tracking_example.py`. In order to use this tracker, two components are needed:

- a configuration specifying:
  - path to object model
  - start pose of the object
- an image handle that:
  - stores camera intrinsics
  - return an image if called, and return None if the user wants to stop tracking

The configuration can be found at the top of `multi_obj_tracking_example.py`; an example image handle object can be found at `rbot_dataset_handle.py`. The example is a replica of the experiments described in the original paper. All the paths corresponds to the local absolute path of the [RBOT dataset](http://cvmr.info/research/RBOT/).

The result video can be found here: https://drive.google.com/file/d/1j3tayDE09-JzOdyNnzkIqhGjPGbYjYZz/view?usp=sharing. Note that this is a test tracking, not an evaluation tracking (that is to say we run normal tracking as we would in real life, with no access to ground truth pose after each iteration, and no resetting).

### Running in Docker

If you built the docker image, you can run the example, as given or modified with:
```
cd fairo/pyrbgt
xhost +local:docker
docker run --rm -it \
  --privileged \
  --gpus=all \
  --network host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=${DISPLAY} \
  -v ~/data:/data \
  -v $(pwd):/pyrbgt \
  pyrbgt \
  python3 /pyrbgt/examples/rbot_example/multi_obj_tracking_example.py
```
Note that this expects the `RBOT_dataset` to be available in `~/data/RBOT_dataset`.

### WIP
- **Renderer is now fixed.**
- Current WIP:
  - A templated rendering-based pose estimator to initialize tracking.
  - Extracting of mask information, and keypoints through UV map.


### Azure Kinect Installation Notes

Follow instructions for setting up the Azure Kinect [here](https://docs.microsoft.com/en-us/azure/Kinect-dk/set-up-azure-kinect-dk).  Summarized below:
- Configure MS package repository:
  - `curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -`
  - `sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod`
  - `sudo apt-get update`

  - Note: The `apt-add-repository` above didn't work on my Facebook desktop on Ubuntu 18.04 (Due to Chef overwriting the main sources.list).  I created a new file /etc/apt/sources.list.d/microsoft.list containing the single line:
    - `deb [arch=amd64] https://packages.microsoft.com/ubuntu/18.04/prod bionic main` - note the added arch restriction to avoid errors related to the repo not supporting i386 arch.
- Install Kinect for Azure tools:
  - `sudo apt install k4a-tools`
- Update to latest firmware
  - Download latest from [this page](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md)
  - Install: `AzureKinectFirmwareTool -u <filename.bin>`
- (Optional) Setup udev rules:
  - Copy [rules](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/scripts/99-k4a.rules) into new file `/etc/udev/rules.d/99-k4a.rules`
  - Disconnect and reconnect device.
- Verify:
  - `AzureKinectFirmwareTool -q` should display firmware info.
  - `k4aviewer` should show camera live view (after clicking 'open' and 'start')
