First of all

- `Camera.py`: windows receives original camera themes
- `Camera2.py`: windows receives compressed camera themes
- `Keyboard_control_velocity.py`: keyboard control of turtlebot robot movement in ubuntu (W A S D)

## rosbridge

The reason I'm looking into this rosbridge is: I FasterRCNN model training on my lab windows computer, but since we're training for reinforcement learning, the turtlebot runs in a gazebo environment on Ubuntu 22.04+Humble in VMware on windows, I need to get the images from the RGB camera or depth camera that runs on the I need to install a RGB camera or depth camera on the robot running in the gazebo world, and get the image by subscribing to the `/camera_sensor/image_raw` topic in the `ros2 topic list`, but the subscription is generally only available in ubuntu, how do I get this live video frame stream from windows? (Because I want to take the video stream frame, pass it to FasterRCNN network and get the prediction result: x,y, conference, width, height, distance of the target frame) I searched online and found rosbridge meets my requirement

### Introduction

[`rosbridge`](https://github.com/RobotWebTools/rosbridge_suite) is an officially supported bridge tool for ROS that allows **non-ROS systems (e.g. Windows) to communicate with ROS nodes** via WebSocket, with the ability to publish and subscribe to topics, You can publish, subscribe to topics, and call services, which is perfectly suited for your “ROS 2 → Windows image transfer” scenario.

### Install

#### Step 1: Install the rosbridge package

```bash
sudo apt update
sudo apt install ros-humble-rosbridge-server
```

#### Step 2: Start the rosbridge WebSocket Service

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

result：

```bash
[INFO] [launch]: All log files can be found below /home/xinyang/.ros/log/2025-04-14-14-31-06-229714-xinyang-virtual-machine-37225
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [rosbridge_websocket-1]: process started with pid [37226]
[INFO] [rosapi_node-2]: process started with pid [37228]
[rosbridge_websocket-1] [INFO] [1744655466.585805888] [rosbridge_websocket]: Rosbridge WebSocket server started on port 9090
```

#### Step 3: Check if rosbridge started successfully

```bash
ros2 topic list | grep rosbridge
```



#### Step 4: Connecting and subscribing to image topics using `roslibpy` on Windows

`roslibpy` has already been tested to receive images successfully. Just make sure:

- The IP address is the IP of your Ubuntu virtual machine (e.g. `192.168.1.xxx`, not localhost).
- The port is `9090`.
- The topic name is `/camera_sensor/image_raw` or `/camera/image_raw/compressed`.



```less
[ Gazebo analog camera ] → /camera_sensor/image_raw
                          |
                          ↓
        [ rosbridge WebSocket @ Ubuntu virtual machine (9090) ]
                          |
                          ↓
        [ Python Programs on Windows (roslibpy) ]
                          |
                          ↓
       [ Faster R-CNN model for real-time image prediction ]
                          |
                          ↓
          [ Output bbox: x, y, conf, w, h, distance ]

```

#### Step 5 Image Compression

I tested using the /camera_sensor/image_raw theme to get an image and transfer it to windows, I wanted to test for latency, so I installed the libraries `keyboard` and `roslibpy` on windows first. 

```bash
pip install roslibpy
pip install keyboard
```

> - Official Documentation: https://roslibpy.readthedocs.io/
> - GitHub source code: https://github.com/gramaziokohler/roslibpy

And use `Keyboard_control_velocity.py` code externally to control the movement of the robot internally, via wasd four buttons.

But I found that after I run the `Keyboard_control_velocity.py` code and check the acquired image, I found that it is lagging, for example, I control the robot to turn left, but the video streaming monitored in windows took 15s before the camera content changed as it should, which is very much out of line with our requirement, because the high latency means that there will be a lag in the image after we recognize it. There is also a lag when we send commands to him, so I would like to find out what the problem is and how to fix it.

The internet suggests using a `/compressed` topic subscription, saying that it reduces the network burden, I learned that this `/compressed` is actually a `/camera_sensor/image_raw/compressed`, which is based on the original `/camera_sensor/image_raw` theme, using a plugin that compresses the original theme and publishes it as a new theme.

```bash
# 1. Install the image_transport plugin
#This plugin automatically generates /compressed topics for all sensor_msgs/Image image topics.

sudo apt update
sudo apt install ros-humble-image-transport-plugins

# 2. Run Gazebo emulation + start camera
#Make sure your robot emulation is started and that the camera 
#is started (i.e. publish /camera/image_raw). Then run it again:

ros2 topic list | grep compressed

# 3. If the front process is not working, try reboot/restart the ubuntu

# 4.If you still don't see /compressed, you can start the forwarding node manually:

ros2 run image_transport republish raw --ros-args -r in:=/camera/image_raw -r out:=/camera/image_raw/compressed

```

<img src="./resources/images/vmware_0KCy19S5xt.png" alt="vmware_0KCy19S5xt" style="zoom:50%;" />

After completing the above steps, you can subscribe like this on Windows:

```python
listener = roslibpy.Topic(client, '/camera/image_raw/compressed', 'sensor_msgs/CompressedImage')
```

Finally, I created a new code `Camera2.py` to show the effect and found it to be real-time

Later, I'll use the PPO agent code to replace this manually typed control

