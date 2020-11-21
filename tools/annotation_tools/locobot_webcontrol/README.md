# Locobot Web Controller
A web app designed to connect to the locobot backend and provide control, view, and annotation capabilities 

The application comes in two parts. One to be run on the devfair cluster, and one to be run on the local machine. This was to get around the limitations of running/developing a node.js server on the devfair, but later they should be integrated.

## How to install (devfair):
First ensure that the locobot server is up and running and that the tests in `minecraft/python/locobot/tests` work (Just ensure that they can connect to the server and send/receive commands).

Next activate the proper conda env:

`conda activate /private/home/apratik/.conda/envs/locobot_env`

Set up port forwarding (using your own devserver)

`ssh -f -J snc-fairjmp101 -L 3000:localhost:3000 tuck@100.97.69.170`

run the server

`cd minecraft/python/locobot`

`python locobot_control_server.py`

Your server should be up and running now, ready to accept commands

The next section will detail how to access the server and start sending commands and getting visuals

## How to install (local machine):

`cd minecraft/annotation_tools/locobot_webcontrol`

`npm install`

`npm start`

As the application starts you should see a live visual feed from the bot, and be able to send commands to the bot.
Currently the available commands are limited to 

`!f[num]` ## moves the bot forward/backward in world space

`!l[num]` ## moves the bot latterly in world space

`!t[num]` ## turns the bot in degrees in world space

TODO: connect the bot to the locobot_agent script and take commands in english.


