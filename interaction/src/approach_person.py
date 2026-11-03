#!/usr/bin/env python3
"""Gesture-confirmed person approach through Nav2; cancel on lost/stale vision."""
import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from g1_interaction_msgs.msg import PersonTracks
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException

from vision_geometry import safe_approaches


class Approach(Node):
    def __init__(self,report,timeout,required_track=None):
        super().__init__('g1_approach_person',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.report_path=Path(report)
        self.deadline=time.monotonic()+timeout
        self.tf=Buffer()
        self.listener=TransformListener(self.tf,self)
        self.map=None
        self.people=None
        self.received=0
        self.odometry=deque(maxlen=50000)
        self.required_track=required_track
        self.confirm_id=None
        self.confirm_stamps=[]
        self.locked=None
        self.target=None
        self.last_seen=0
        self.phase='waiting_for_raised_hand'
        self.pending=None
        self.nav_handle=None
        self.nav_result=None
        self.cancel_future=None
        self.cancel_deadline=0.
        self.candidates=iter(())
        self.settle_since=None
        self.done=False
        self.cancel_requested=False
        self.planner=ActionClient(self,ComputePathToPose,'/compute_path_to_pose')
        self.nav=ActionClient(self,NavigateToPose,'/navigate_to_pose')
        self.lifecycle={name:self.create_client(GetState,'/'+name+'/get_state')
                        for name in ('planner_server','bt_navigator','collision_monitor')}
        self.lifecycle_pending={}
        self.lifecycle_sent={}
        self.lifecycle_active=set()
        self.lifecycle_checked=0.
        self.status=self.create_publisher(String,'/interaction/approach_status',10)
        self.create_subscription(PersonTracks,'/interaction/people',self.on_people,10)
        self.create_subscription(OccupancyGrid,'/map',lambda m:setattr(self,'map',m),
             QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry,'/g1_odometry_publisher/odom',self.on_odom,10)
        self.timer=self.create_timer(.2,self.tick)
        self.data={'passed':False,'input':'RGB-D YOLO pose + ByteTrack','truth_fed_to_algorithm':False,
                   'navigation':'Nav2 NavigateToPose','events':[]}

    def on_odom(self,msg):
        p=msg.pose.pose.position
        self.odometry.append((p.x,p.y,msg.twist.twist.linear.x,msg.twist.twist.angular.z))

    def on_people(self,msg):
        if msg.header.frame_id!='map':
            return
        age=(self.get_clock().now()-Time.from_msg(msg.header.stamp)).nanoseconds/1e9
        if not 0<=age<=1.2:
            return
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        self.people=msg
        self.received=time.monotonic()
        if self.locked is not None:
            for person in msg.people:
                if person.track_id==self.locked:
                    new=np.array([person.position.x,person.position.y])
                    if np.linalg.norm(new-self.target)>.65:
                        self.abort('confirmed person moved; reacquisition required')
                        return
                    self.target=.8*self.target+.2*new
                    self.last_seen=time.monotonic()
            return
        raised=[p for p in msg.people if p.raised_hand and p.confidence>=.45
                and (self.required_track is None or p.track_id==self.required_track)]
        # Ambiguous gestures do not select the nearest arbitrary person.
        if len(raised)!=1:
            self.confirm_id=None
            self.confirm_stamps=[]
            return
        person=raised[0]
        if person.track_id!=self.confirm_id:
            self.confirm_id=person.track_id
            self.confirm_stamps=[]
        if not self.confirm_stamps or stamp>self.confirm_stamps[-1]:
            if self.confirm_stamps and stamp-self.confirm_stamps[-1]>.8:
                self.confirm_stamps=[]
            self.confirm_stamps.append(stamp)
        self.confirm_stamps=self.confirm_stamps[-20:]
        if (len(self.confirm_stamps)>=4 and stamp-self.confirm_stamps[0]>=.6
                and self.map and len(self.lifecycle_active)==len(self.lifecycle)):
            self.locked=person.track_id
            self.target=np.array([person.position.x,person.position.y])
            self.last_seen=time.monotonic()
            try:
                robot=self.robot()
            except TransformException:
                self.locked=None
                return
            m=self.map
            q=m.info.origin.orientation
            origin_yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            self.candidates=iter(safe_approaches(self.target,robot,
                np.asarray(m.data).reshape(m.info.height,m.info.width),m.info.resolution,
                [m.info.origin.position.x,m.info.origin.position.y,origin_yaw]))
            self.phase='planning'
            self.data.update(track_id=self.locked,confirmed_person_map_xy=self.target.tolist(),
                             confirmation_frames=len(self.confirm_stamps))
            self.event('gesture_confirmed')

    def robot(self):
        t=self.tf.lookup_transform('map','base_footprint',Time()).transform.translation
        return np.array([t.x,t.y])

    def event(self,text):
        self.data['events'].append({'event':text,'sim_time':self.get_clock().now().nanoseconds/1e9})
        self.get_logger().info(text)

    def pose(self,xyyaw):
        p=PoseStamped()
        p.header.frame_id='map'
        p.header.stamp=self.get_clock().now().to_msg()
        p.pose.position.x,p.pose.position.y=float(xyyaw[0]),float(xyyaw[1])
        p.pose.orientation.z=math.sin(xyyaw[2]/2)
        p.pose.orientation.w=math.cos(xyyaw[2]/2)
        return p

    def abort(self,reason):
        if self.done or self.cancel_requested:
            return
        self.data['failure']=reason
        self.event(reason)
        self.cancel_requested=True
        if self.nav_handle and self.nav_handle.accepted:
            self.cancel_future=self.nav_handle.cancel_goal_async()
            self.cancel_deadline=time.monotonic()+8
            self.phase='cancelling'
            return
        # An in-flight goal acceptance must be handled before the node exits.
        if self.phase not in ('sending_navigation','turn_acceptance','room_nav_acceptance'):
            self.finish(False)

    def finish(self,passed):
        self.done=True
        self.data.update(passed=passed,phase='completed' if passed else 'failed')
        self.data['odom_distance_m']=sum(math.hypot(b[0]-a[0],b[1]-a[1])
                                         for a,b in zip(self.odometry,list(self.odometry)[1:]))
        self.report_path.parent.mkdir(parents=True,exist_ok=True)
        self.report_path.write_text(json.dumps(self.data,indent=2)+'\n')
        self.status.publish(String(data=json.dumps(self.data)))

    def tick(self):
        if self.done:
            return
        now=time.monotonic()
        if self.phase=='cancelling':
            if (self.nav_result is not None and self.nav_result.done()) or now>self.cancel_deadline:
                self.data['cancellation_result_received']=bool(self.nav_result and self.nav_result.done())
                self.finish(False)
            return
        for name,future in list(self.lifecycle_pending.items()):
            if future.done():
                state=future.result()
                if state and state.current_state.id==State.PRIMARY_STATE_ACTIVE:
                    self.lifecycle_active.add(name)
                else:
                    self.lifecycle_active.discard(name)
                del self.lifecycle_pending[name]
                self.lifecycle_sent.pop(name,None)
            elif now-self.lifecycle_sent.get(name,now)>3:
                # A service request made during server discovery can be lost.
                # Retry with a fresh request instead of waiting indefinitely.
                self.lifecycle[name].remove_pending_request(future)
                del self.lifecycle_pending[name]
                self.lifecycle_sent.pop(name,None)
        if now-self.lifecycle_checked>1.:
            for name,client in self.lifecycle.items():
                if name not in self.lifecycle_pending and client.service_is_ready():
                    self.lifecycle_pending[name]=client.call_async(GetState.Request())
                    self.lifecycle_sent[name]=now
            self.lifecycle_checked=now
        if now>self.deadline:
            self.abort('mission timeout')
            return
        if self.locked is not None and now-self.last_seen>2.0:
            self.abort('visual track lost or stale; cancel navigation')
            return
        if self.phase=='planning' and self.pending is None and self.planner.server_is_ready():
            try:
                self.goal=next(self.candidates)
            except StopIteration:
                self.abort('no safe reachable approach endpoint')
                return
            request=ComputePathToPose.Goal(goal=self.pose(self.goal),planner_id='GridBased')
            self.pending=self.planner.send_goal_async(request)
            self.phase='plan_acceptance'
        elif self.phase=='plan_acceptance' and self.pending.done():
            h=self.pending.result()
            if not h.accepted:
                self.phase='planning';self.pending=None
            else:
                self.pending=h.get_result_async();self.phase='plan_result'
        elif self.phase=='plan_result' and self.pending.done():
            result=self.pending.result()
            self.pending=None
            if result.status!=GoalStatus.STATUS_SUCCEEDED or len(result.result.path.poses)<2:
                self.phase='planning'
                return
            if not self.nav.server_is_ready():
                self.phase='planning'
                return
            self.data['approach_goal']=list(self.goal)
            self.data['path_poses']=len(result.result.path.poses)
            self.event('nav2_path_accepted')
            self.pending=self.nav.send_goal_async(NavigateToPose.Goal(pose=self.pose(self.goal)))
            self.phase='sending_navigation'
        elif self.phase=='sending_navigation' and self.pending.done():
            self.nav_handle=self.pending.result()
            if self.cancel_requested:
                if self.nav_handle.accepted:
                    self.nav_result=self.nav_handle.get_result_async()
                    self.cancel_future=self.nav_handle.cancel_goal_async()
                    self.cancel_deadline=time.monotonic()+8
                    self.phase='cancelling'
                else:
                    self.finish(False)
            elif not self.nav_handle.accepted:
                self.abort('Nav2 rejected approach')
            else:
                self.nav_result=self.nav_handle.get_result_async()
                self.pending=self.nav_result;self.phase='walking'
                self.event('walking_to_confirmed_person')
        elif self.phase=='walking' and self.pending.done():
            result=self.pending.result()
            if result.status!=GoalStatus.STATUS_SUCCEEDED:
                self.abort('Nav2 approach failed: '+str(result.status))
            else:
                self.phase='checking_stop'
                self.settle_since=None
        elif self.phase=='checking_stop':
            try:
                robot=self.robot()
            except TransformException:
                return
            distance=float(np.linalg.norm(robot-self.target))
            stopped=self.odometry and abs(self.odometry[-1][2])<.08 and abs(self.odometry[-1][3])<.08
            if not 1.0<=distance<=1.95:
                self.abort('final person separation outside accepted range')
            elif stopped:
                if self.settle_since is None:
                    self.settle_since=now
                if now-self.settle_since>=3:
                    self.data.update(final_person_separation_m=distance,final_robot_map_xy=robot.tolist(),
                                     final_person_map_xy=self.target.tolist(),stop_observation_s=3)
                    self.event('stopped_beside_confirmed_person')
                    self.finish(True)
            else:
                self.settle_since=None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',default='/workspace/logs/person-approach.json')
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--track-id',type=int)
    args=parser.parse_args()
    rclpy.init()
    node=Approach(args.report,args.timeout,args.track_id)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node,timeout_sec=.1)
    except KeyboardInterrupt:
        node.abort('interrupted')
        until=time.monotonic()+9
        while not node.done and rclpy.ok() and time.monotonic()<until:
            rclpy.spin_once(node,timeout_sec=.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if node.data['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
