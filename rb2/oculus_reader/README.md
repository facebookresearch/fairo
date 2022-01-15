# oculus_reader

This repository provides the tool to read the position and pressed button from the Oculus Quest device.

Oculus reader consits of two elements: python script which receives the readings from the APK and the APK itself. Currently the pose of the controllers and pressed buttons are transfered from the APK. This behavior can be extended using provided APK [source code](app_source).

## Clone the repository

To pull the APK correctly, Git LFS has to be configured before cloning the repository. The installation is described here https://git-lfs.github.com. On Ubuntu follow these steps:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install # has to be run only once on a single user account
```

Now you can clone this repository either with HTTPS or SSH.

If you decide to use the HTTPS protocol, you can reduce the number of authentification prompts when pushing/pulling caused by Git LFS with:

```bash
git config lfs.https://github.com/rail-berkeley/oculus_reader.git/info/lfs.locksverify false
```

This command has to be run from the repository workspace.

## Setup of the ADB

[ADB](https://developer.android.com/studio/command-line/adb) is required for the communication between Oculus Quest and the python reader script.

To install ADB on Ubuntu run:

```
sudo apt install android-tools-adb
```

On other systems follow the steps from the 'app_source' folder.

### Set up of a new Oculus Quest device

1. Determine your Oculus Quest account name:
If you haven’t used Oculus Quest before, start it and follow the steps to create your profile and get yourself started. Otherwise follow these steps to find out your username:
    1. Go to: [https://www.oculus.com/](https://www.oculus.com/) 
    2. Log in to account:
    ![image_0](https://user-images.githubusercontent.com/14967831/106832581-c7288f00-6646-11eb-91e0-3b74e81a58ba.png)
    3. After logging in **select your profile again** in top right corner and select **‘Profile’**
    ![image_1](https://user-images.githubusercontent.com/14967831/106832585-c859bc00-6646-11eb-9a3d-3a55f844ee37.png)
    4. You will be able to see your username on the following screen:
    ![image_2](https://user-images.githubusercontent.com/14967831/106832678-f7702d80-6646-11eb-823e-1001d6bffe01.png)
2. Enable Oculus Quest development mode:
    1. If you belong to RAIL, inform me (Jedrzej Orbik) that you need to join the development organization. This is required to activate debugging mode on the device. Otherwise create your own organization <https://developer.oculus.com/manage/organizations/create/> and fill in the appropriate informaiton.
    2. Turn on the device you want to use for development.
    3. Open the Oculus app on your phone and then go to **Settings**.
    4. Tap the device and then go to **More Settings** > **Developer Mode**.
    5. Turn on the **Developer Mode** toggle.
    6. Connect your device to your computer using a USB-C cable and then wear the device.
    7. Accept **Allow USB Debugging** and **Always allow from this computer** when prompted to on the device.  
        ![image_3](https://user-images.githubusercontent.com/14967831/104061507-048d2e80-51f9-11eb-8327-7917f6a1ab60.png)  
    8. (Windows only) Install the Oculus ADB Drivers
        1. Download [the zip file containing the driver](https://developer.oculus.com/downloads/package/oculus-adb-drivers/).
        2. Unzip the file.
        3. Right-click on the .inf file and select **Install**.

## How to run the code

If you intend to use the precompiled APK with the predefined behavior, where the position and the pressed buttons are transferred, please follow the steps from the [scripts folder](oculus_reader/README.md). If you plan to extend the app, please read the README from the [app_source folder](app_source/README.md).

### Communication using the USB cable (easier to set up)

1. Connect Oculus Quest to PC with USB cable. This is required to establish the connection.
2. Run the exemplary file: `python oculus_reader/reader.py`
3. (optionally) If ROS is installed on your machine, the current transformation can be visualized using the script [visualize_oculus_transforms.py](oculus_reader/visualize_oculus_transforms.py).


### Communication over the network (more portable)

1. Make sure that Oculus Quest is connected to the same network as the computer.
2. Connect Oculus Quest to PC with USB cable. This is required to establish the connection.
3. Put on the headset and allow the permission as requested.
4. Verify that a device is visible with: `adb devices`. The expected output:  
`List of devices attached`  
`    ce0551e7                device`
5. Check the IP address of the headset:  
    `adb shell ip route`  
    Expected output:  
    `10.0.30.0/19 dev wlan0  proto kernel  scope link  **src **10.0.32.101`
6. Read the IP address of the device standing after `**src`.
7. Provide the IP address when creating OculusReader object.
8. Run the exemplary file: `python oculus_reader/reader.py`
9. (optionally) If ROS is installed on your machine, the current transformation can be visualized using the script [visualize_oculus_transforms.py](oculus_reader/visualize_oculus_transforms.py).
