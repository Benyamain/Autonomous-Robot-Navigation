#!/usr/bin/env python3

import math, os, time, subprocess
from typing import List
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
import torch, torch.nn as nn
from torch.distributions import Normal

K_ANGLE = 0.3

def teacher_controller(rel_x: float, rel_y: float):
    ang = math.atan2(rel_y, rel_x)
    w   = K_ANGLE * ang
    v   = (-w/(3*math.pi)+0.2) if w>0 else (w/(3*math.pi)+0.2)
    return np.clip(v,0,0.22), np.clip(w,-1.5,1.5)

class Actor(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(6,128),nn.ReLU(),nn.Linear(128,2)); self.std=0.5
    def forward(self,x): return Normal(self.net(x), torch.full((2,), self.std))
class Critic(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(6,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,x): return self.net(x)

class Buffer:
    def __init__(self): self.clear()
    def clear(self):
        self.states:List[torch.Tensor]=[]; self.actions=[]; self.logp=[]; self.rews=[]; self.dones=[]; self.vals=[]

class ResPPO(Node):
    def __init__(self):
        super().__init__('residual_ppo')
        # ROS
        self.pub = self.create_publisher(Twist,'/cmd_vel',10)
        self.create_subscription(JointState ,'/_joint_states',self.cb_joint,10)
        self.create_subscription(ModelStates,'/gazebo/model_states',self.cb_model,10)
        self.create_subscription(PoseArray  ,'/object_detections',self.cb_det ,10)
        self.timer=self.create_timer(0.5,self.step)
        # 状态
        self.robot_x=self.robot_y=0.; self.goal_x=self.goal_y=0.; self.rel_x=self.rel_y=0.; self.target_dist=float('inf'); self.vel=[0.,0.]
        # PPO
        self.actor, self.critic = Actor(), Critic();
        self.optA=torch.optim.Adam(self.actor.parameters(),3e-4)
        self.optC=torch.optim.Adam(self.critic.parameters(),1e-3)
        self.buf=Buffer(); self.ep=self.step_ct=0; self.max_steps=200
        # EMA
        self.alpha=0.2; self.v_ema=self.w_ema=0.; self.last_w=0.

    def cb_joint(self,msg): self.vel=msg.velocity[:2]
    def cb_model(self,msg):
        try:
            idx=msg.name.index('turtlebot3_burger'); p=msg.pose[idx].position
            self.robot_x,self.robot_y=p.x,p.y
        except ValueError: pass
    def cb_det(self,msg):
        if not msg.poses: return
        p=msg.poses[0]; self.goal_x,self.goal_y=p.position.x,p.position.y
        self.rel_x=self.goal_x-self.robot_x; self.rel_y=self.goal_y-self.robot_y
        self.target_dist=math.hypot(self.rel_x,self.rel_y)

    def get_state(self):
        return torch.tensor([self.target_dist,*self.vel,1.0,self.step_ct/self.max_steps,1.],dtype=torch.float32)
    def ema(self,new,old): return self.alpha*new+(1-self.alpha)*old

    def step(self):
        if math.isinf(self.target_dist): return
        self.step_ct+=1
        s=self.get_state(); dist=self.actor(s); a=dist.sample(); lp=dist.log_prob(a).sum(); vpred=self.critic(s)
        # teacher
        v_t,w_t = teacher_controller(self.rel_x,self.rel_y)
        # residuals
        dv = 0.05*torch.tanh(a[0]).item(); dw = 0.30*torch.tanh(a[1]).item()
        v = np.clip(v_t+dv,0,0.22); w = np.clip(w_t+dw,-1.5,1.5)
        # publish (EMA)
        self.v_ema,self.w_ema=self.ema(v,self.v_ema),self.ema(w,self.w_ema)
        t=Twist(); t.linear.x=self.v_ema; t.angular.z=self.w_ema; self.pub.publish(t)
        # reward
        r = 30*self.v_ema - 0.1 - 4*abs(w) - 2*abs(w-self.last_w); self.last_w=w
        done=False
        if self.target_dist<0.2: r+=1500; done=True
        elif self.step_ct>=self.max_steps: done=True
        # buffer
        self.buf.states.append(s); self.buf.actions.append(a.detach()); self.buf.logp.append(lp.detach());
        self.buf.rews.append(r); self.buf.dones.append(done); self.buf.vals.append(vpred)
        if done: self.update(); self.reset_ep()

    def update(self):
        returns,adv=[],[]; gae=0.; next_v=0.
        for r,d,v in zip(self.buf.rews[::-1],self.buf.dones[::-1],self.buf.vals[::-1]):
            delta=r+0.99*next_v*(1-d)-v.item(); gae=delta+0.99*0.95*(1-d)*gae
            adv.insert(0,gae); returns.insert(0,gae+v.item()); next_v=v.item()
        adv=torch.tensor(adv,dtype=torch.float32); returns=torch.tensor(returns,dtype=torch.float32)
        adv=(adv-adv.mean())/(adv.std()+1e-8); states=torch.stack(self.buf.states).float(); acts=torch.stack(self.buf.actions).float(); old_lp=torch.stack(self.buf.logp).float()
        for _ in range(4):
            d=self.actor(states); lp=d.log_prob(acts).sum(1); ratio=torch.exp(lp-old_lp)
            aloss=-torch.min(ratio*adv,torch.clamp(ratio,0.8,1.2)*adv).mean(); vpred=self.critic(states).squeeze(); closs=nn.functional.mse_loss(vpred,returns)
            loss=aloss+0.5*closs-0.01*d.entropy().sum(1).mean(); self.optA.zero_grad(); self.optC.zero_grad(); loss.backward(); self.optA.step(); self.optC.step()

    def reset_ep(self):
        self.get_logger().info(f"EP {self.ep:03d} | R={sum(self.buf.rews):.1f} | avg_v={np.mean([t.linear.x for t in []]):.2f}")
        self.buf.clear(); self.ep+=1; self.step_ct=0; self.target_dist=float('inf'); self.v_ema=self.w_ema=self.last_w=0.


def main():
    os.environ['ROS_DOMAIN_ID']='30'
    subprocess.run(['ros2','service','call','/reset_simulation','std_srvs/srv/Empty'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    time.sleep(2)
    rclpy.init(); node=ResPPO(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()
