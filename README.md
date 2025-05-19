# :robot: Integration of CNN and PINN with Reinforcement Learning for Autonomous Robot Navigation and Object Detection in a ROS-Gazebo Environment :robot:
This paper tackles the complexities of mobile robot
navigation in dynamic industrial environments, where traditional approaches often falter due to unpredictable obstacles
and evolving conditions. We propose an integrated framework
that synergizes vision-based perception, physics-informed control,
and reinforcement learning (RL) to achieve effective navigation.
Deployed on the TurtleBot3 platform within a ROS2 Humble
and Gazebo Classic simulation, our framework combines a
Convolutional Neural Network (CNN) for object detection and
distance estimation, a Physics-Informed Neural Network (PINN)
for wheel dynamics control, and Proximal Policy Optimization
(PPO) for decision-making. This approach enables the robot to
detect target objects and navigate efficiently in dynamic settings.
Our experimental results demonstrate the framework’s effectiveness, achieving real-time control at 30 Hz with a 6-dimensional
state space and 2-dimensional action space. The system maintains
stable navigation with precise torque control (±0.1 N·m) while
supporting multi-threaded execution for simultaneous wheel
control. Performance evaluation shows 93.69% recall and 86.89%
precision in reaching target locations, achieved through optimized training strategies, while maintaining a minimum safe
distance of 0.7 m from obstacles. The integrated CNN-PINN-PPO architecture demonstrates robust adaptation to varying
surface conditions and dynamic obstacles, though the PPO agent
faces training stability challenges. Simulation results highlight
the framework’s potential to manage complex scenarios, with
challenges such as the sim-to-real gap, computational demands,
and training convergence remaining for future improvement.

### Dataset Link: 
- https://udmercy0-my.sharepoint.com/:f:/g/personal/zhangxi24_udmercy_edu/EggDTcSJJeJOjv5RSyIxa3MB4MXuBeLqY2m39IiqLGuQcg?e=wutRdC
