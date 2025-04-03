import torch
import torch.nn as nn
import numpy as np
import threading
from dataclasses import dataclass
from typing import Tuple, Optional
from rclpy.node import Node
from threading import Lock

@dataclass
class WheelState:
    """Physical state of the wheels"""
    vel_left: float
    vel_right: float
    pos_left: float
    pos_right: float
    torque_left: float
    torque_right: float
    timestamp: float

class WheelDynamicsPINN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # load config
        self.wheel_radius = config['wheel_radius']
        self.wheel_base = config['wheel_base']
        self.max_torque = config['max_torque']
        self.wheel_inertia = config['wheel_inertia']
        self.friction_coeff = config['friction_coeff']
        
        # neural network for dynamics prediction
        layers = []
        prev_dim = 8  # [left_state, right_state, torques, dt]
        for dim in config['hidden_dims']:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 4))
        
        self.dynamics_net = nn.Sequential(*layers) # all layers in a sequence
        
        # thread locks for wheel control
        self.left_lock = threading.Lock()
        self.right_lock = threading.Lock()
        
        # ROS2 rate control
        self.control_rate = 1.0 / config.get('control_frequency', 100.0)  # default 100Hz?
        
        # thread-safe management
        self.state_lock = Lock()
        self.current_state = WheelState(
            vel_left=0.0, vel_right=0.0,
            pos_left=0.0, pos_right=0.0,
            torque_left=0.0, torque_right=0.0,
            timestamp=0.0
        )
        
    def compute_kinematic_loss(self, state: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Differential drive kinematic constraints"""
        # extract velocities
        v_l, v_r = pred[:, 0], pred[:, 1]
        
        # linear and angular velocity
        v = (v_r + v_l) * self.wheel_radius / 2
        w = (v_r - v_l) * self.wheel_radius / self.wheel_base
        
        # kinematic constraint: v = (v_r + v_l)/2
        kinematic_residual = torch.abs(v - (state[:, 0] + state[:, 1])/2)
        return kinematic_residual.mean()
    
    def compute_dynamic_loss(self, state: torch.Tensor, pred: torch.Tensor, 
                           torques: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Physical dynamics constraints"""
        # torque-acceleration relationship: τ = I * α
        accel_left = (pred[:, 0] - state[:, 0]) / dt
        accel_right = (pred[:, 1] - state[:, 1]) / dt
        
        # dynamic equations
        torque_residual_left = torques[:, 0] - (self.wheel_inertia * accel_left + 
                                               self.friction_coeff * state[:, 0])
        torque_residual_right = torques[:, 1] - (self.wheel_inertia * accel_right + 
                                                self.friction_coeff * state[:, 1])
        
        return (torch.abs(torque_residual_left) + torch.abs(torque_residual_right)).mean()
    
    def forward(self, state: torch.Tensor, torques: torch.Tensor, 
                dt: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with physics-informed loss
        Args:
            state: [batch_size, 4] - [vel_left, vel_right, pos_left, pos_right]
            torques: [batch_size, 2] - [torque_left, torque_right]
            dt: [batch_size] - timestep
        """
        # combine inputs
        x = torch.cat([state, torques, dt.unsqueeze(-1)], dim=1)
        predictions = self.dynamics_net(x)
        
        if self.training:
            # physics-informed losses
            kinematic_loss = self.compute_kinematic_loss(state, predictions)
            dynamic_loss = self.compute_dynamic_loss(state, predictions, torques, dt)
            
            # data-driven loss (MSE with actual next state when available)
            physics_loss = kinematic_loss + dynamic_loss
            
            return predictions, physics_loss
        
        return predictions, None
    
    def apply_torque(self, wheel: str, torque: float, duration: float, node: Node) -> None:
        """
        Apply torque using ROS2 timing
        Args:
            wheel: 'left' or 'right'
            torque: torque value in N⋅m
            duration: time to apply torque in seconds
            node: ROS2 node for timing
        """
        lock = self.left_lock if wheel == 'left' else self.right_lock
        with lock:
            # clamp torque to physical limits
            torque = np.clip(torque, -self.max_torque, self.max_torque)
            
            # calculate steps based on control frequency
            control_period = 1.0 / node.get_parameter('control_frequency').value
            steps = int(duration / control_period)
            
            # create torque profiles
            torque_profile = np.linspace(0, torque, steps)
            reverse_profile = np.linspace(torque, 0, steps//2)
            
            # use ROS2 Rate for timing
            rate = node.create_rate(1.0 / control_period)
            
            # Apply increasing torque
            for t in torque_profile:
                with self.state_lock:
                    if wheel == 'left':
                        self.current_state.torque_left = float(t)
                    else:
                        self.current_state.torque_right = float(t)
                rate.sleep()
            
            # Apply decreasing torque for smooth stop
            for t in reverse_profile:
                with self.state_lock:
                    if wheel == 'left':
                        self.current_state.torque_left = float(t)
                    else:
                        self.current_state.torque_right = float(t)
                rate.sleep()

    def predict_next_state(self, current_state: WheelState, 
                          target_vel_left: float, target_vel_right: float) -> WheelState:
        """Predict next wheel state based on current state and target velocities"""
        # prepare input tensor
        state = torch.tensor([
            [current_state.vel_left, current_state.vel_right,
             current_state.pos_left, current_state.pos_right]
        ])
        torques = torch.tensor([
            [current_state.torque_left, current_state.torque_right]
        ])
        dt = torch.tensor([0.01])  # 10ms prediction steps
        
        # get prediction
        with torch.no_grad():
            pred, _ = self.forward(state, torques, dt)
        
        # update state
        next_state = WheelState(
            vel_left=float(pred[0, 0]),
            vel_right=float(pred[0, 1]),
            pos_left=float(pred[0, 2]),
            pos_right=float(pred[0, 3]),
            torque_left=current_state.torque_left,
            torque_right=current_state.torque_right,
            timestamp=current_state.timestamp + 0.01
        )
        
        return next_state 