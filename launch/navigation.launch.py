from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_nav',  # some package name that we will name.
            executable='ppo_agent',   # some entry point that we will name.
            name='ppo_agent',
            parameters=[{
                'max_episode_steps': 500,
                'min_target_distance': 0.5,
                'max_target_distance': 5.0,
                'collision_threshold': 0.3,
                'max_linear_velocity': 0.5,
                'max_angular_velocity': 1.0,
                'control_frequency': 100.0,
                'wheel_base': 0.160,
                'wheel_radius': 0.033,
                'max_torque': 0.1,
                'wheel_inertia': 0.001,
                'friction_coeff': 0.1
            }]
        )
    ]) 