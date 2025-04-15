import roslibpy
import base64
import numpy as np
import cv2
import time

client = roslibpy.Ros(host='172.18.18.202', port=9090)
client.run()

def callback(msg):
    try:
        data = msg['data']
        decoded = base64.b64decode(data)
        np_arr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        cv2.imshow("Compressed Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            listener.unsubscribe()
            client.terminate()
            cv2.destroyAllWindows()
    except Exception as e:
        print("Image processing failure:", e)

listener = roslibpy.Topic(client, '/camera_sensor/image_raw/compressed', 'sensor_msgs/CompressedImage')
listener.subscribe(callback)

print("Receiving compressed image...")

# replace run_forever
try:
    while client.is_connected:
        time.sleep(0.1)
except KeyboardInterrupt:
    listener.unsubscribe()
    client.terminate()
    cv2.destroyAllWindows()
    print("Interrupt Exit")
