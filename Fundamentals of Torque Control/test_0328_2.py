import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import ApplyJointEffort
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from rclpy.parameter import Parameter
import threading
import time
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy

def float_to_duration(value):
    sec = int(value)
    nanosec = int((value - sec) * 1e9)
    return sec, nanosec

def spin_until_future_complete(node, future):
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.spin_until_future_complete(future)
    executor.shutdown()

class ClockListener(Node):
    def __init__(self):
        super().__init__('clock_listener')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.current_sim_time = 0.0

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(Clock, '/clock', self.clock_callback, qos_profile)

    def clock_callback(self, msg):
        self.current_sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def get_sim_time(self):
        return self.current_sim_time
    
class ApplyEffortClient(Node):
    def __init__(self, joint_name):
        super().__init__(f'apply_effort_client_{joint_name}')
        self.joint_name = joint_name
        self.cli = self.create_client(ApplyJointEffort, '/apply_joint_effort')
        self.joint_velocity = None
        self.joint_state_event = threading.Event()

        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

    def joint_state_callback(self, msg):
        if self.joint_name in msg.name:
            idx = msg.name.index(self.joint_name)
            velocity = msg.velocity[idx]
            self.joint_velocity = velocity

            # 判断速度是否足够小
            if abs(velocity) < 0.001:
                self.get_logger().info(f"{self.joint_name} velocity near zero: {velocity}")
                self.joint_state_event.set()

    def wait_until_stopped(self, timeout=2.0):
        self.get_logger().info(f"Waiting for {self.joint_name} to stop...")
        return self.joint_state_event.wait(timeout=timeout)

    def send_effort_request(self, joint_name, effort, start_time_sec, duration_sec):
        request = ApplyJointEffort.Request()
        request.joint_name = joint_name
        request.effort = float(effort)

        sec, nanosec = float_to_duration(start_time_sec)
        request.start_time.sec = sec
        request.start_time.nanosec = nanosec

        sec, nanosec = float_to_duration(duration_sec)
        request.duration.sec = sec
        request.duration.nanosec = nanosec

        future = self.cli.call_async(request)
        return future

def spin_clock_node(clock_node):
    executor = SingleThreadedExecutor()
    executor.add_node(clock_node)
    while rclpy.ok():
        executor.spin_once(timeout_sec=0.1)

def apply_effort(joint_name, effort, delay, duration, clock_node):
    node = ApplyEffortClient(joint_name)

    while clock_node.get_sim_time() == 0.0 and rclpy.ok():
        time.sleep(0.1)

    current_sim_time = clock_node.get_sim_time()
    scheduled_start = current_sim_time + delay

    # Step 1: 正向力矩
    future = node.send_effort_request(joint_name, effort, scheduled_start, duration)
    spin_until_future_complete(node, future)

    if future.result() is not None:
        node.get_logger().info(f"{joint_name} response: success={future.result().success}, "
                               f"message={future.result().status_message}")

    reverse_effort = -effort * 0.7
    reverse_start = scheduled_start + duration
    reverse_duration = 0.5
    reverse_future = node.send_effort_request(joint_name, reverse_effort, reverse_start, reverse_duration)
    spin_until_future_complete(node, reverse_future)

    # Step 3: 等待速度接近 0
    stopped = node.wait_until_stopped(timeout=2.0)

    if stopped:
        node.get_logger().info(f"{joint_name} has stopped based on /joint_states")
    else:
        node.get_logger().warn(f"{joint_name} did not stop in expected time")

    # Step 4: 最终施加 0 力矩
    stop_future = node.send_effort_request(joint_name, 0.0, 0.0, 1.0)
    spin_until_future_complete(node, stop_future)

    if stop_future.result() is not None:
        node.get_logger().info(f"{joint_name} fully stopped: success={stop_future.result().success}, "
                               f"message={stop_future.result().status_message}")
        
    node.destroy_node()

def main(args=None):
    rclpy.init(args=args)

    # 创建 clock listener 并在后台安全运行
    clock_node = ClockListener()
    spin_thread = threading.Thread(target=spin_clock_node, args=(clock_node,), daemon=True)
    spin_thread.start()

    delay_left = 2
    delay_right = 2
    duration_left = 1
    duration_right = 1
    effort_left = 0.25
    effort_right = 0.25

    thread_left = threading.Thread(target=apply_effort, args=(
        'wheel_left_joint', effort_left, delay_left, duration_left, clock_node
    ))
    thread_right = threading.Thread(target=apply_effort, args=(
        'wheel_right_joint', effort_right, delay_right, duration_right, clock_node
    ))

    thread_left.start()
    thread_right.start()

    thread_left.join()
    thread_right.join()

    clock_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
