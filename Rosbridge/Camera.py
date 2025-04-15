import roslibpy
import numpy as np
import cv2
import base64

client = roslibpy.Ros(host='172.18.18.202', port=9090)
client.run()

def callback(msg):
    try:
        width = msg['width']
        height = msg['height']
        encoding = msg['encoding']
        data = msg['data']

        # Decode base64 image data
        decoded_bytes = base64.b64decode(data)
        img_np = np.frombuffer(decoded_bytes, dtype=np.uint8)

        # Reshape to image format
        img_np = img_np.reshape((height, width, 3))

        # Convert to BGR if the encoding is rgb8
        if encoding == 'rgb8':
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif encoding != 'bgr8':
            print(f"Unsupported encoding format: {encoding}")
            return

        cv2.imshow("ROS Image", img_np)
        cv2.waitKey(1)

    except Exception as e:
        print("Failed to decode image:", e)

listener = roslibpy.Topic(client, '/camera_sensor/image_raw', 'sensor_msgs/Image')
listener.subscribe(callback)

print("Started receiving image stream...")

try:
    while client.is_connected:
        pass
except KeyboardInterrupt:
    listener.unsubscribe()
    client.terminate()
    print("Terminated")
