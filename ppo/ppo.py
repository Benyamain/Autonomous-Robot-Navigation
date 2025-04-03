import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import JointState
from collections import deque
from pinn.wheel_dynamics_pinn import WheelDynamicsPINN, WheelState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future

class PPOMemory:
    def __init__(self):
        # store experiences for training
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def store(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, action_std_init):
        super(ActorNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim), # linearize the previous layer to the next layer
                nn.ReLU() # activation func
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, action_dim)) # linearize the previous layer to the action dimension
        
        self.actor = nn.Sequential(*layers)
        self.action_std = nn.Parameter(torch.ones(action_dim) * action_std_init) # parametrize the action std
    
    def forward(self, state):
        action_mean = self.actor(state)
        action_dist = Normal(action_mean, self.action_std)
        return action_dist

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # output value estimate
        )
    
    def forward(self, state):
        return self.critic(state)

# NOTE: we need to figure out the the hyperparameters for the PPO agent!
class PPOAgent(Node):
    def __init__(self):
        super().__init__('ppo_agent')
        
        self.declare_parameter('control_frequency', 100.0)
        
        self.state_dim = 6  # [target_distance, min_obstacle_distance, velocities[2], target_in_view, normalized_steps]
        self.action_dim = 2  # [linear_velocity, angular_velocity]
        
        self.declare_parameters(
            'ppo_agent', # this node name is used in navigation.launch.py
            [
                # env parameters
                ('max_episode_steps', None),
                ('min_target_distance', None),
                ('max_target_distance', None),
                ('collision_threshold', None),
                ('max_linear_velocity', None),
                ('max_angular_velocity', None),
                
                # ppo hyperparameters
                ('gamma', 0.99),
                ('gae_lambda', 0.95),
                ('clip_epsilon', 0.2),
                ('actor_lr', 0.0003),
                ('critic_lr', 0.0003),
                ('pinn_lr', 0.0001),
                
                # network arch
                ('actor_hidden_dims', [256, 128]), # 256 neurons in the first layer, 128 neurons in the second layer
                ('critic_hidden_dims', [256, 128]), # same as above
                ('pinn_hidden_dims', [64, 32]), # you get the idea
                ('action_std_init', 0.1),
                
                # reward thresholds
                ('wheel_diff_threshold', None),
                ('wheel_sync_threshold', None),
                ('angular_vel_threshold', None),
                
                # turtlebot specific
                ('wheel_base', None), # figure it out?
                
                # pinn parameters
                ('wheel_radius', 0.033),
                ('max_torque', 0.1),
                ('wheel_inertia', 0.001),
                ('friction_coeff', 0.1),
            ]
        )
        
        # loading parameters
        self.max_episode_steps = self.get_parameter('max_episode_steps').value
        self.min_target_distance = self.get_parameter('min_target_distance').value
        self.max_target_distance = self.get_parameter('max_target_distance').value
        self.collision_threshold = self.get_parameter('collision_threshold').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        
        self.gamma = self.get_parameter('gamma').value
        self.gae_lambda = self.get_parameter('gae_lambda').value
        self.clip_epsilon = self.get_parameter('clip_epsilon').value
        
        self.wheel_diff_threshold = self.get_parameter('wheel_diff_threshold').value
        self.wheel_sync_threshold = self.get_parameter('wheel_sync_threshold').value
        self.angular_vel_threshold = self.get_parameter('angular_vel_threshold').value
        
        self.wheel_base = self.get_parameter('wheel_base').value
        
        # initialize pinn
        pinn_config = {
            'wheel_radius': self.get_parameter('wheel_radius').value,
            'wheel_base': self.wheel_base,
            'max_torque': self.get_parameter('max_torque').value,
            'wheel_inertia': self.get_parameter('wheel_inertia').value,
            'friction_coeff': self.get_parameter('friction_coeff').value,
            'hidden_dims': self.get_parameter('pinn_hidden_dims').value,
            'control_frequency': self.get_parameter('control_frequency').value
        }
        
        # initialize networks
        self.actor = ActorNetwork(
            self.state_dim, 
            self.action_dim,
            self.get_parameter('actor_hidden_dims').value,
            self.get_parameter('action_std_init').value
        )
        self.critic = CriticNetwork(self.state_dim)
        self.wheel_dynamics = WheelDynamicsPINN(pinn_config)
        
        # state variables
        self.steps = 0
        self.current_detections = None
        self.current_velocities = None
        self.target_in_view = False
        self.target_distance = float('inf')
        self.min_obstacle_distance = float('inf')
        self.detection_buffer = deque(maxlen=5)
        
        # ROS2 communication
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_state_sub = self.create_subscription(
            JointState, 
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.detection_sub = self.create_subscription(
            PoseArray,
            '/object_detections',
            self.detection_callback,
            10
        )
        
        # initialize optimizer
        self.memory = PPOMemory()
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # track wheel state
        self.current_wheel_state = WheelState(
            vel_left=0.0,
            vel_right=0.0,
            pos_left=0.0,
            pos_right=0.0,
            torque_left=0.0,
            torque_right=0.0,
            timestamp=0.0
        )
        
        # add callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()
        
    def joint_state_callback(self, msg):
        """process wheel joint states"""
        self.current_velocities = np.array(msg.velocity)
        # update wheel state
        self.current_wheel_state.vel_left = msg.velocity[0]
        self.current_wheel_state.vel_right = msg.velocity[1]
        self.current_wheel_state.pos_left = msg.position[0]
        self.current_wheel_state.pos_right = msg.position[1]
        self.current_wheel_state.timestamp = self.get_clock().now().nanoseconds / 1e9

    # NOTE: based the impl. from your guys' code?
    def detection_callback(self, msg):
        """process object detections concurrentfrom CNN"""
        detections = []
        for pose in msg.poses:
            detection = np.array([
                pose.position.x,
                pose.position.y,
                pose.orientation.x,  # class_id
                pose.orientation.y,  # confidence
                pose.orientation.z,  # width
                pose.orientation.w,  # height
                pose.position.z,     # yaw
            ])
            detections.append(detection)
        
        self.detection_buffer.append(detections)
        self._update_target_info()
        
    def _update_target_info(self):
        """update target-related information"""
        if not self.detection_buffer:
            self.target_in_view = False
            return
            
        recent_detections = np.mean(self.detection_buffer, axis=0)
        target_detections = [d for d in recent_detections if d[2] == 1]  # class_id == 1
        
        if target_detections:
            self.target_in_view = True
            self.target_distance = min(np.sqrt(d[0]**2 + d[1]**2) for d in target_detections)
            
            obstacles = [d for d in recent_detections if d[2] != 1]
            self.min_obstacle_distance = float('inf') if not obstacles else \
                min(np.sqrt(d[0]**2 + d[1]**2) for d in obstacles)
        else:
            self.target_in_view = False
    
    def get_state(self):
        """construct state vector"""
        if self.current_velocities is None:
            return None
            
        state = np.array([
            self.target_distance,
            self.min_obstacle_distance,
            self.current_velocities[0],  # linear velocity
            self.current_velocities[1],  # angular velocity
            float(self.target_in_view),
            self.steps / self.max_episode_steps,  # normalized step count
        ])
        return torch.FloatTensor(state)
    
    # NOTE: the rewards defined are not set in stone, we can change them!
    def compute_reward(self):
        """calculate reward based on current state"""
        reward = 0.0
        
        # target-related rewards
        if self.target_in_view:
            reward += 1.0
            if self.min_target_distance < self.target_distance < self.max_target_distance:
                reward += 2.0 * (self.max_target_distance - self.target_distance) / self.max_target_distance
        
        # safety penalties
        if self.min_obstacle_distance < self.collision_threshold:
            reward -= 5.0
        
        # smooth motion reward
        if self.current_velocities is not None:
            angular_vel = abs(self.current_velocities[1])
            if angular_vel > 0.5:
                reward -= 0.1 * angular_vel
        
        # add wheel dynamics rewards
        if self.current_wheel_state is not None:
            # reward smooth wheel transitions
            wheel_diff = np.abs(self.current_wheel_state.vel_left - self.current_wheel_state.vel_right)
            if wheel_diff > 0.2:  # threshold for smooth transition
                reward -= 0.5 * wheel_diff
                
            # reward synchronized wheel behavior
            wheel_sync = np.abs(self.current_wheel_state.vel_left - self.current_wheel_state.vel_right)
            if wheel_sync > 0.3:  # threshold for synchronization
                reward -= 0.3 * wheel_sync
        
        return reward
    
    def select_action(self, state):
        """select action using current policy"""
        with torch.no_grad():
            action_dist = self.actor(state)
            action = action_dist.sample()
            
            # convert linear/angular to wheel velocities
            v = action[0]  # linear velocity
            w = action[1]  # angular velocity
            
            # differential drive kinematics
            v_l = v - (w * self.wheel_base / 2)
            v_r = v + (w * self.wheel_base / 2)
            
            # use pinn to predict next state
            next_state = self.wheel_dynamics.predict_next_state(
                current_state=self.current_wheel_state,
                target_vel_left=v_l,
                target_vel_right=v_r
            )
            
            # apply torques concurrently
            left_future = Future()
            right_future = Future()
            
            def apply_left_torque():
                self.wheel_dynamics.apply_torque('left', next_state.torque_left, 0.1, self)
                left_future.set_result(None)
            
            def apply_right_torque():
                self.wheel_dynamics.apply_torque('right', next_state.torque_right, 0.1, self)
                right_future.set_result(None)
            
            # schedule torque applications
            self.executor.create_task(apply_left_torque, callback_group=self.callback_group)
            self.executor.create_task(apply_right_torque, callback_group=self.callback_group)
            
            # wait for both to complete
            rclpy.spin_until_future_complete(self, left_future)
            rclpy.spin_until_future_complete(self, right_future)
            
            # use predictions to adjust actions
            actual_v = (next_state.vel_left + next_state.vel_right) / 2
            actual_w = (next_state.vel_right - next_state.vel_left) / self.wheel_base
            
            action = torch.tensor([actual_v, actual_w])
        
        # clip actions to safe ranges (woooo)
        action[0] = torch.clamp(action[0], -self.max_linear_velocity, self.max_linear_velocity)
        action[1] = torch.clamp(action[1], -self.max_angular_velocity, self.max_angular_velocity)
        
        # publish velocity command
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        
        return action
    
    def update_policy(self, batch_size=64):
        """Update policy using PPO"""
        states = torch.FloatTensor(np.array(self.memory.states))
        actions = torch.FloatTensor(np.array(self.memory.actions))
        old_probs = torch.FloatTensor(np.array(self.memory.probs))
        values = torch.FloatTensor(np.array(self.memory.vals))
        rewards = torch.FloatTensor(np.array(self.memory.rewards))
        dones = torch.FloatTensor(np.array(self.memory.dones))
        
        # compute advantages
        advantages = torch.zeros_like(rewards) # advantages are an estimate of the relative value for a selected action
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            last_gae = advantages[t]
        
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # ppo update
        for _ in range(5):  # 5 epochs
            for batch_start in range(0, len(states), batch_size):
                batch_end = batch_start + batch_size
                state_batch = states[batch_start:batch_end]
                action_batch = actions[batch_start:batch_end]
                old_prob_batch = old_probs[batch_start:batch_end]
                advantage_batch = advantages[batch_start:batch_end]
                
                # actor loss
                action_dist = self.actor(state_batch)
                new_probs = action_dist.log_prob(action_batch)
                ratio = torch.exp(new_probs - old_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantage_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # critic loss
                value_batch = self.critic(state_batch)
                critic_loss = nn.MSELoss()(value_batch, rewards[batch_start:batch_end])
                
                # update networks
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
        
        self.memory.clear()
    
    def is_done(self):
        """Check if episode should terminate"""
        return (
            self.steps >= self.max_episode_steps or
            self.target_distance < self.min_target_distance or
            self.min_obstacle_distance < self.collision_threshold or
            (not self.target_in_view and self.steps > 50)
        )
    
    def reset(self):
        """Reset environment for new episode"""
        self.steps = 0
        self.target_in_view = False
        self.target_distance = float('inf')
        self.min_obstacle_distance = float('inf')
        self.detection_buffer.clear()
        
        # stop the robot
        stop_cmd = Twist()
        self.vel_pub.publish(stop_cmd)
        
        return self.get_state()

def main(args=None):
    rclpy.init(args=args)
    agent = PPOAgent()
    
    # NOTE: you guys think this is a good way to do it?
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
