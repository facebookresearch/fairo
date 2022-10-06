# iPhone Reader

Thin wrapper around [Record3D](https://record3d.app/) to easily retrieve sensor information from an iPhone including:
- RGB image
- Depth image
- camera pose estimation

## Requirements
- CMake (can be installed via Conda)

## Installation
```sh
pip install .
```

## Usage

### Device setup

1. Get an iPhone with lidar (iPhone 13 Pro or iPhone 14 Pro)
1. Connect phone to computer via USB
1. Download the Record3D app from the App Store (premium version required for USB streaming) and launch the app
1. Set "Settings > Live RGBD Video Streaming" to "USB"
1. Click record button to start streaming

### API examples

Polling
```py
from iphone_reader import iPhoneReader

reader = iPhoneReader()
reader.start()

frame = reader.wait_for_frame()
```

Callback
```py
from iphone_reader import iPhoneReader

def print_frame(frame):
    print(frame)

reader = iPhoneReader()
reader.start(frame_callback=print_frame)
```