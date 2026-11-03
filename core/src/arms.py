#!/usr/bin/env python3
"""Reuse Grove's upper-body trajectory controller; never command legs or base pose."""
import argparse,json,math,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from controller_manager_msgs.srv import ListControllers,SwitchController
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from arm_motion import ArmMotion,JOINTS


class RelaxedArms(Node):
    def __init__(self,report):
        super().__init__('xiaobei_workbench_arms',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.report=Path(report);self.states={};self.velocities={};self.cmd=Twist();self.cmd_at=-1e6
        self.motion=None;self.last_tick=None;self.state_at=-1e6
        self.active=False;self.pending=None;self.phase='list';self.start=None;self.target=None
        self.stats={'controller':'existing arm_trajectory_controller','leg_policy_unchanged':True,'activated':False,'samples':0,'max_shoulder_swing_rad':0.0}
        self.create_subscription(JointState,'/joint_states',self.joints,10)
        self.create_subscription(Twist,'/cmd_vel',self.command,10)
        self.list=self.create_client(ListControllers,'/controller_manager/list_controllers')
        self.switch=self.create_client(SwitchController,'/controller_manager/switch_controller')
        self.pub=self.create_publisher(JointTrajectory,'/arm_trajectory_controller/joint_trajectory',2)
        self.create_timer(.02,self.tick)
    def joints(self,msg):
        self.states.update(zip(msg.name,msg.position))
        self.velocities.update(zip(msg.name,msg.velocity))
        self.state_at=self.get_clock().now().nanoseconds/1e9
        self.stats['last_measured_arm_positions']={j:self.states[j] for j in JOINTS if j in self.states}
        if math.hypot(self.cmd.linear.x,self.cmd.linear.y)>.08:
            ranges=self.stats.setdefault('walking_joint_ranges_rad',{})
            for j in JOINTS+['left_hip_pitch_joint','right_hip_pitch_joint']:
                if j in self.states:
                    lo,hi=ranges.get(j,[self.states[j]]*2);v=self.states[j];ranges[j]=[min(lo,v),max(hi,v)]
    def command(self,msg):self.cmd=msg;self.cmd_at=self.get_clock().now().nanoseconds/1e9
    def write(self):
        self.report.parent.mkdir(parents=True,exist_ok=True);self.report.write_text(json.dumps(self.stats,indent=2)+'\n')
    def tick(self):
        if not all(j in self.states for j in JOINTS):return
        if not self.active:
            if self.pending:
                if not self.pending.done():
                    if time.monotonic()-self.sent>3:
                        self.pending.cancel();self.pending=None;self.phase='list'
                    return
                response=self.pending.result();self.pending=None
                if self.phase=='switch':
                    self.active=bool(response and response.ok);self.stats['activated']=self.active;self.write()
                    self.phase='list'
                    if self.active:self.get_logger().info('Relaxed arms: existing trajectory controller active; legs unchanged')
                    return
                states={x.name:x.state for x in response.controller} if response else {}
                if states.get('arm_trajectory_controller')=='active':self.active=True;self.stats['activated']=True;self.write();return
                if states.get('arm_freeze_controller')!='active' or states.get('arm_trajectory_controller')!='inactive':return
                if not self.switch.service_is_ready():return
                req=SwitchController.Request(activate_controllers=['arm_trajectory_controller'],deactivate_controllers=['arm_freeze_controller'],strictness=SwitchController.Request.STRICT,activate_asap=True,timeout=Duration(sec=2))
                self.pending=self.switch.call_async(req);self.phase='switch';self.sent=time.monotonic();return
            if self.list.service_is_ready():self.pending=self.list.call_async(ListControllers.Request());self.sent=time.monotonic();self.phase='list'
            return
        now=self.get_clock().now().nanoseconds/1e9
        if now-self.state_at>.3:return
        if self.motion is None:self.motion=ArmMotion(self.states)
        dt=.02 if self.last_tick is None else now-self.last_tick;self.last_tick=now
        valid=0<=now-self.cmd_at<.3
        positions,velocities=self.motion.update(self.states,self.velocities,
            self.cmd.linear.x if valid else 0.,self.cmd.angular.z if valid else 0.,dt)
        msg=JointTrajectory(joint_names=JOINTS)
        # The existing controller requires a zero-velocity terminal point.
        # Refresh before this bounded stopping tail, retaining continuous
        # positions/velocities while also stopping safely if updates cease.
        msg.points=[JointTrajectoryPoint(positions=positions,velocities=velocities,
            time_from_start=Duration(nanosec=40000000)),
            JointTrajectoryPoint(positions=[p+.02*v for p,v in zip(positions,velocities)],
                velocities=[0.]*len(JOINTS),time_from_start=Duration(nanosec=80000000))]
        self.pub.publish(msg);self.stats['samples']+=1
        self.stats['max_shoulder_swing_rad']=max(self.stats['max_shoulder_swing_rad'],abs(self.motion.swing))
        self.stats.update(command_rate_hz=50,phase_source='measured hip positions and velocities',
                          max_arm_joint_speed_rad_s=self.motion.max_speed,
                          max_arm_joint_acceleration_rad_s2=self.motion.max_acceleration)
        if self.stats['samples']%100==0:self.write()


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args();rclpy.init();n=RelaxedArms(a.report)
    try:rclpy.spin(n)
    except KeyboardInterrupt:pass
    finally:n.write();n.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
