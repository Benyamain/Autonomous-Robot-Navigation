#!/usr/bin/env python3

import math, os, time, subprocess
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Optional
from nav_msgs.msg import Odometry

try:
    from tf_transformations import euler_from_quaternion, quaternion_from_euler
except ImportError: 
    from transforms3d.euler import quat2euler as _q2e
    def euler_from_quaternion(q):
        return _q2e([q[3], q[0], q[1], q[2]])

# # ─────────────── Teacher controller ───────────────
# K_ANGLE = 0.3

# def teacher_controller(rel_x: float, rel_y: float):
#     angle = math.atan2(rel_y, rel_x)
#     omega = K_ANGLE * angle
#     v = (-omega / (3 * math.pi) + 0.2) if omega > 0 else (omega / (3 * math.pi) + 0.2)
#     return np.clip(v, 0.0, 0.22), np.clip(omega, -1.5, 1.5)
ROS_DOMAIN_ID = "30"  
COORDINATES = np.array( 
    [
        # [0.0, 0.0],
        [goal_x, goal_y],
    ],
    dtype=float,
)
LOOP_HZ = 10.0                    
K_ANGLE = 0.3                          
PERCENT_FACTOR = 15.0 / 100.0           
RANGE_MAX = PERCENT_FACTOR 

def double_reset_simulation() -> None:
    subprocess.run(
        ["ros2", "service", "call", "/reset_simulation", "std_srvs/srv/Empty"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)
    subprocess.run(
        ["ros2", "service", "call", "/reset_simulation", "std_srvs/srv/Empty"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def double_reset_simulation() -> None:
    subprocess.run(
        ["ros2", "service", "call", "/reset_simulation", "std_srvs/srv/Empty"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)
    subprocess.run(
        ["ros2", "service", "call", "/reset_simulation", "std_srvs/srv/Empty"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

class Turtlebot332(Node):
    def __init__(self):
        super().__init__("turtlebot332")

        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sub_odom = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10
        )
        self.sub_model = self.create_subscription(
            ModelStates, "/gazebo/model_states", self._model_cb, 10
        )

        self.odom_msg: Optional[Odometry] = None
        self.model_msg: Optional[ModelStates] = None

        # self.x_plot, self.y_plot = [], []
        # self.xm, self.ym = [], []
        # self.plot_t, self.plot_theta = [], []
        # self.plot_linearvel, self.plot_angularvel = [], []
        # self.PL = 0.0
        # self.start_time = time.time()

        self.rate_dt = 1.0 / LOOP_HZ

    def _odom_cb(self, msg: Odometry):
        self.odom_msg = msg

    def _model_cb(self, msg: ModelStates):
        self.model_msg = msg

    def odom_measure(self) -> tuple[float, float]:
        while self.odom_msg is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        p = self.odom_msg.pose.pose.position
        return p.x, p.y

    def run(self):
        while self.odom_msg is None or self.model_msg is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        robot_name = "turtlebot3_burger"
        try:
            robot_index = self.model_msg.name.index(robot_name)
        except ValueError:
            self.get_logger().error(f'"{robot_name}" not in /model_states')
            return

        velmsg = Twist()
        num_edges = COORDINATES.shape[0]

        for nS in range(1, num_edges + 1):
            prex = prey = math.nan
            len_dist = float("inf")

            while len_dist > RANGE_MAX:
                rclpy.spin_once(self, timeout_sec=0.0)
                cur_xy = np.array(self.odom_measure())
                target_xy = COORDINATES[nS % num_edges]

                len_dist = float(np.linalg.norm(cur_xy - target_xy))

                q = self.odom_msg.pose.pose.orientation
                yaw_current = euler_from_quaternion(
                    [q.x, q.y, q.z, q.w]
                )[2]

                # t_now = time.time() - self.start_time
                # self.plot_t.append(t_now)
                # self.plot_theta.append(yaw_current)

                dx, dy = (target_xy - cur_xy).tolist()
                if dx == 0.0:
                    thdr = math.copysign(math.pi / 2, dy)
                else:
                    slope = dy / dx
                    thd = math.atan(slope)
                    if slope >= 0:
                        theta_cand = [thd, -math.pi + thd]
                    else:
                        theta_cand = [thd, math.pi + thd]

                    if slope == 0:
                        thdr = theta_cand[0] if dx > 0 else theta_cand[1]
                    elif slope > 0 and dx > 0:
                        thdr = next(t for t in theta_cand if t > 0)
                    elif slope < 0 and dx < 0:
                        thdr = next(t for t in theta_cand if t > 0)
                    elif slope > 0 and dx < 0:
                        thdr = next(t for t in theta_cand if t < 0)
                    elif slope < 0 and dx > 0:
                        thdr = next(t for t in theta_cand if t < 0)
                    else:
                        thdr = thd

                err_angle = wrap_to_pi(thdr - yaw_current)
                omega_real = K_ANGLE * err_angle
                if omega_real > 0:
                    velocity_real = -(1.0 / (3 * math.pi)) * omega_real + 0.2
                else:
                    velocity_real = (1.0 / (3 * math.pi)) * omega_real + 0.2

                velmsg.linear.x = velocity_real
                velmsg.angular.z = omega_real
                self.pub_vel.publish(velmsg)

                # if not math.isnan(prex):
                #     self.PL += math.hypot(cur_xy[0] - prex, cur_xy[1] - prey)
                # prex, prey = cur_xy

                # self.x_plot.append(cur_xy[0])
                # self.y_plot.append(cur_xy[1])

                # model_pos = self.model_msg.pose[robot_index].position
                # self.xm.append(model_pos.x)
                # self.ym.append(model_pos.y)

                # self.plot_linearvel.append(velocity_real)
                # self.plot_angularvel.append(omega_real)

                time.sleep(self.rate_dt)

        velmsg.linear.x = 0.0
        velmsg.angular.z = 0.0
        self.pub_vel.publish(velmsg)


class GoalListener(Node):
    def __init__(self):
        super().__init__('goal_listener')

        # 订阅发布的假目标检测
        self.create_subscription(
            PoseArray,
            '/object_detections',          # 话题名要与发布端保持一致
            self.detection_callback,
            10                              # QoS 深度，默认即可
        )

    # ----------------- 回调：处理目标 -----------------
    def detection_callback(self, msg: PoseArray):
        if not msg.poses:
            self.get_logger().warn('收到空的 PoseArray')
            return

        # 这里只取第一个目标。如有多目标可遍历 msg.poses
        pose = msg.poses[0]

        # 地图坐标系下的目标位置
        goal_x = pose.position.x
        goal_y = pose.position.y

        # orientation 四元数里塞的元数据（与发布端约定的编码保持一致）
        class_id   = pose.orientation.x   # 目标类别
        confidence = pose.orientation.y   # 置信度
        bbox_w     = pose.orientation.z   # 边框宽
        bbox_h     = pose.orientation.w   # 边框高

        # self.get_logger().info(
        #     f"收到目标：map(x={goal_x:.2f}, y={goal_y:.2f}) "
        #     f"class={int(class_id)}, conf={confidence:.2f}, "
        #     f"bbox=({bbox_w:.2f},{bbox_h:.2f})"
        # )


# ─────────────── Networks ───────────────
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 2))
        self.std = torch.tensor([0.2, 0.2])
    def forward(self, x):
        return Normal(self.net(x), self.std)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        return self.net(x)

class Buffer:
    def __init__(self):
        self.clear()
    def clear(self):
        self.states, self.actions, self.logp = [], [], []
        self.rews, self.dones, self.vals     = [], [], []

# ─────────────── Agent ───────────────
class PPOAgent(Node):
    def __init__(self):
        super().__init__('ppo_teacher_blend')
        # Pub/Sub
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(JointState, '/joint_states', self.cb_joint, 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.cb_model, 10)
        self.create_subscription(PoseArray, '/object_detections', self.cb_det, 10)
        self.timer = self.create_timer(0.5, self.step)

        # State
        self.robot_x = self.robot_y = 0.0
        self.goal_x  = self.goal_y  = 0.0
        self.rel_x   = self.rel_y   = 0.0
        self.target_dist = float('inf'); self.target_in_view = False
        self.vel = [0.0, 0.0]

        # PPO
        self.actor, self.critic = Actor(), Critic()
        self.optA = torch.optim.Adam(self.actor.parameters(), 3e-4)
        self.optC = torch.optim.Adam(self.critic.parameters(), 1e-3)
        self.buf  = Buffer()

        # Hyper
        self.step_ct = 0; self.ep = 0; self.max_steps = 200
        self.warmup_eps = 100; self.init_blend = 0.7
        # EMA
        self.alpha = 0.2; self.v_ema = self.w_ema = 0.0; self.last_ang = 0.0

    # ---------- Callbacks ----------
    def cb_joint(self, msg):
        self.vel = msg.velocity[:2]

    def cb_model(self, msg):
        try:
            idx = msg.name.index('turtlebot3_burger')
            p = msg.pose[idx].position
            self.robot_x, self.robot_y = p.x, p.y
        except ValueError:
            pass

    def cb_det(self, msg):
        if not msg.poses: return
        p = msg.poses[0]
        self.goal_x, self.goal_y = p.position.x, p.position.y
        self.rel_x = self.goal_x - self.robot_x
        self.rel_y = self.goal_y - self.robot_y
        self.target_dist = math.hypot(self.rel_x, self.rel_y)
        self.target_in_view = True

    # ---------- Helpers ----------
    def get_state(self):
        return torch.tensor([
            self.target_dist, *self.vel, float(self.target_in_view), self.step_ct / self.max_steps, 1.0
        ], dtype=torch.float32)

    def ema(self, new, old):
        return self.alpha * new + (1 - self.alpha) * old

    # ---------- Main step ----------
    def step(self):
        if not self.target_in_view:
            return
        self.step_ct += 1

        s = self.get_state()
        dist = self.actor(s)
        a    = dist.sample()
        logp = dist.log_prob(a).sum()
        val  = self.critic(s)

        # Teacher & PPO actions
        v_t, w_t = Turtlebot332(velmsg.linear.x, velmsg.angular.z)
        v_p = (torch.tanh(a[0]).item() + 1) * 0.11
        w_p = torch.tanh(a[1]).item() * 1.5

        if self.ep < self.warmup_eps:
            blend = 1.0 - (1.0 - self.init_blend) * (self.ep / self.warmup_eps)
        else:
            blend = 0.0
        v = blend * v_t + (1 - blend) * v_p
        w = blend * w_t + (1 - blend) * w_p

        # Publish
        self.v_ema, self.w_ema = self.ema(v, self.v_ema), self.ema(w, self.w_ema)
        tw = Twist(); tw.linear.x = self.v_ema; tw.angular.z = self.w_ema
        self.pub.publish(tw)

        # Reward: 时间惩罚 + 速度奖励 + 旋转惩罚
        r = 25 * self.v_ema - 0.2 - 4 * abs(w) - 2 * abs(w - self.last_ang)
        self.last_ang = w
        done = False
        if self.target_dist < 0.2:
            r += 1500; done = True
        elif self.step_ct >= self.max_steps:
            done = True

        # Buffer store
        self.buf.states.append(s)
        self.buf.actions.append(a.detach())
        self.buf.logp.append(logp.detach())
        self.buf.rews.append(float(r))
        self.buf.dones.append(done)
        self.buf.vals.append(val)

        if done:
            self.update_policy(); self.reset_ep(blend)

    # ---------- PPO update ----------
    def update_policy(self):
        # ---- compute returns & advantages (GAE) ----
        returns, adv = [], []
        gae, next_v = 0.0, 0.0
        for r, d, v in zip(self.buf.rews[::-1], self.buf.dones[::-1], self.buf.vals[::-1]):
            delta = r + 0.99 * next_v * (1 - d) - v.item()
            gae   = delta + 0.99 * 0.95 * (1 - d) * gae
            adv.insert(0, gae)
            returns.insert(0, gae + v.item())
            next_v = v.item()
        adv     = torch.tensor(adv,     dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        states  = torch.stack(self.buf.states).float()
        actions = torch.stack(self.buf.actions).float()
        old_lp  = torch.stack(self.buf.logp ).float()

        for _ in range(4):  # PPO epochs
            dist = self.actor(states)
            lp   = dist.log_prob(actions).sum(1)
            ratio = torch.exp(lp - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
            actor_loss  = -torch.min(surr1, surr2).mean()
            value_pred  = self.critic(states).squeeze()
            critic_loss = nn.functional.mse_loss(value_pred, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().sum(1).mean()

            self.optA.zero_grad(); self.optC.zero_grad();
            loss.backward(); self.optA.step(); self.optC.step()

    # ---------- Episode reset ----------
    def reset_ep(self, blend):
        self.get_logger().info(
            f"Episode {self.ep:03d} | reward={sum(self.buf.rews):.1f} | blend_end={blend:.2f}"
        )
        self.buf.clear(); self.step_ct = 0; self.ep += 1; self.target_in_view = False
        # reset ema so next episode 不会突然大跳变
        self.v_ema = self.w_ema = 0.0

# ───────────────────────── main ─────────────────────────

def main():
    os.environ['ROS_DOMAIN_ID'] = ROS_DOMAIN_ID
    double_reset_simulation()
    rclpy.init()
    node = Turtlebot332()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    node = PPOAgent()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
