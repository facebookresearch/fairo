# Epson Moverio BT-35e Camera and Sensor broadcast server

This is a fork of the Moverio SDK Sample app that is modified to broadcast the Moverio sensors and camera over HTTP.
I wrote this primarily to hook the Moverio glasses onto Droidlet running on a server on the same local network.

Instructions:

1. Compile the app with Android Studio, and run it on the Android Tablet attached to the Moverio BT-35E glasses
2. In the app, open the camera view, and touch the "Open Camera" button and the "Start Capture" button


Then, the tablet starts broadcasting two end point:

`http://[tablet ip address]/camera` starts serving the camera image
`http://[tablet ip address]/sensors` starts serving all the sensor readings as a csv file

The camera image is currently hardcoded to 640 x 480 @ 15fps, because that's what I wanted.
