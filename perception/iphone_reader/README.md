# iPhone Reader

Thin wrapper around [Record3D](https://record3d.app/) to easily retrieve sensor information from an iPhone including:
- RGB image
- Depth image
- Camera pose estimation

## Requirements
- CMake (can be installed via Conda)

## Installation
```sh
pip install git+https://github.com/facebookresearch/fairo.git@main#subdirectory=perception/iphone_reader
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

### Example of output `frame`
```
R3dFrame(pose_mat=array([[ 9.99832715e-01,  9.81316959e-03, -1.54351623e-02,
         4.50305249e-03],
       [-9.58856584e-03,  9.99848040e-01,  1.45587435e-02,
         2.91995958e-03],
       [ 1.55756842e-02, -1.44083069e-02,  9.99774874e-01,
         2.41576625e-02],
       [ 4.96068065e-16,  2.53610415e-17, -2.25026389e-16,
         1.00000000e+00]]), pose_pos=array([0.00450305, 0.00291996, 0.02415766]), pose_quat=array([-0.00724226, -0.00775324, -0.00485076,  0.99993195]), color_img=array([[[117, 202, 189],
        [105, 187, 175],
        [102, 184, 172],
        ...,
        [164, 125,  80],
        [160, 122,  76],
        [158, 117,  70]],

       [[149, 233, 222],
        [130, 215, 202],
        [ 98, 180, 168],
        ...,
        [156, 117,  72],
        [152, 113,  66],
        [153, 114,  67]],

       [[119, 201, 189],
        [132, 214, 203],
        [113, 196, 184],
        ...,
        [154, 116,  70],
        [158, 119,  74],
        [160, 121,  76]],

       ...,

       [[120,  61,  41],
        [128,  71,  50],
        [130,  74,  53],
        ...,
        [134,  77,  56],
        [129,  72,  49],
        [123,  65,  42]],

       [[122,  63,  43],
        [129,  73,  52],
        [132,  75,  54],
        ...,
        [131,  74,  53],
        [128,  71,  50],
        [125,  68,  47]],

       [[127,  68,  48],
        [129,  72,  51],
        [131,  75,  54],
        ...,
        [121,  64,  43],
        [120,  62,  39],
        [122,  63,  43]]], dtype=uint8), depth_img=array([[0.9892578 , 1.0419922 , 0.9946289 , ..., 0.38476562, 0.3684082 ,
        0.33544922],
       [1.0322266 , 0.9941406 , 1.0097656 , ..., 0.37524414, 0.3791504 ,
        0.37426758],
       [1.0273438 , 1.0009766 , 1.0253906 , ..., 0.3803711 , 0.38183594,
        0.38476562],
       ...,
       [0.6201172 , 0.6411133 , 0.6308594 , ..., 0.2705078 , 0.2746582 ,
        0.28125   ],
       [0.62841797, 0.65625   , 0.63916016, ..., 0.27416992, 0.2800293 ,
        0.28076172],
       [0.6333008 , 0.6147461 , 0.640625  , ..., 0.26831055, 0.25585938,
        0.26953125]], dtype=float32))
```

## Notes / Troubleshooting

1. Ensure "Higher quality LiDAR recording" is off else there will be a ~5 second latency
1. Under certain conditions setting "USB Streaming RGB quality" to too high will result in corrupted RGB images