import roslibpy
import keyboard
import time

# The ip address here is the IPv4 address from ubuntu's ifconfig
client = roslibpy.Ros(host='172.18.18.202', port=9090) 
client.run()

publisher = roslibpy.Topic(client, '/cmd_vel', 'geometry_msgs/Twist')

# Default linear and angular velocities
LINEAR_SPEED = 0.2
ANGULAR_SPEED = 1.0

def send_cmd(linear=0.0, angular=0.0):
    msg = roslibpy.Message({
        'linear': {'x': linear, 'y': 0.0, 'z': 0.0},
        'angular': {'x': 0.0, 'y': 0.0, 'z': angular}
    })
    publisher.publish(msg)

print("Use WASD to control the robot (press Q to exit)")
try:
    while True:
        if keyboard.is_pressed('w'):
            send_cmd(LINEAR_SPEED, 0.0)
        elif keyboard.is_pressed('s'):
            send_cmd(-LINEAR_SPEED, 0.0)
        elif keyboard.is_pressed('a'):
            send_cmd(0.0, ANGULAR_SPEED)
        elif keyboard.is_pressed('d'):
            send_cmd(0.0, -ANGULAR_SPEED)
        elif keyboard.is_pressed('q'):
            send_cmd(0.0, 0.0)
            print("Exit control")
            break
        else:
            send_cmd(0.0, 0.0)

        time.sleep(0.1)

except KeyboardInterrupt:
    send_cmd(0.0, 0.0)
    print("Disconnect")

publisher.unadvertise()
client.terminate()
