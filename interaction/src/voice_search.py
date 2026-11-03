#!/usr/bin/env python3
"""Audio-guided search: measured bearing -> observed rooms -> multimodal approach.

This node never imports scene/acoustic fixtures and has no simulator truth input.
Canopy resolves room goals; Nav2 owns all planning, collision checks and motion.
"""
import argparse
import json
import math
import time
from pathlib import Path
import cv2
import numpy as np
import rclpy
from canopy_msgs.msg import RoomArray
from canopy_msgs.srv import GetApproachPose
from g1_interaction_msgs.msg import CallDirection
from nav2_msgs.action import NavigateToPose, Spin
from action_msgs.msg import GoalStatus
from lifecycle_msgs.srv import GetState
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from tf2_ros import TransformException
from approach_person import Approach
from scipy.ndimage import distance_transform_edt


def wrap(angle):
    return math.atan2(math.sin(angle),math.cos(angle))


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))


def bearing_error(point, call):
    delta=np.asarray(point)-np.asarray(call['origin'])
    return abs(wrap(math.atan2(delta[1],delta[0])-call['map_azimuth']))


class VoiceSearch(Approach):
    def __init__(self,report,timeout,turn_only=False):
        super().__init__(report,timeout)
        self.phase='waiting_for_name'
        self.turn_only=turn_only
        self.approaching=False
        self.calls=[]
        self.rooms=None
        self.room_id=None
        self.visited=set()
        self.listen_since=0.
        self.observe_wall=time.monotonic()
        self.visual_history={}
        self.visual_packet=None
        self.turn_rounds=0
        self.visual_turns=0
        self.spin_client=ActionClient(self,Spin,'/spin')
        self.room_client=self.create_client(GetApproachPose,'/canopy/get_approach_pose')
        self.lifecycle['behavior_server']=self.create_client(GetState,'/behavior_server/get_state')
        self.create_subscription(CallDirection,'/interaction/call_direction',self.on_call,10)
        self.create_subscription(RoomArray,'/canopy/rooms',self.on_rooms,
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.data.update(input='audio-only sherpa-onnx/SRP-PHAT + RGB-D YOLO/ByteTrack',
            mode='turn_only' if turn_only else 'voice_guided_person_search',
            truth_fed_to_algorithm=False,room_goals_source='live Canopy observed-map GetApproachPose',
            calls=self.calls,visited_rooms=[],rejected_calls=0,ambiguity_events=0,
            multimodal_confirmation=False)

    def seconds(self):
        return self.get_clock().now().nanoseconds/1e9

    def event(self,text):
        super().event(text)
        self.report_path.parent.mkdir(parents=True,exist_ok=True)
        self.report_path.with_suffix('.progress.json').write_text(json.dumps(
            {**self.data,'phase':self.phase},indent=2)+'\n')

    def on_rooms(self,msg):
        if msg.header.frame_id=='map':
            self.rooms=msg

    def on_call(self,msg):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if (not msg.valid or msg.keyword!='小北' or msg.header.frame_id!='microphone_array'
                or not math.isfinite(msg.azimuth_rad) or not 0<=self.seconds()-stamp<=2):
            self.data['rejected_calls']+=1
            return
        if self.calls and stamp<=self.calls[-1]['stamp']:
            return
        try:
            # Measurement-time map transform, never today's pose applied to old audio.
            transform=self.tf.lookup_transform('map',msg.header.frame_id,Time.from_msg(msg.header.stamp))
        except TransformException:
            self.data['rejected_calls']+=1
            return
        t=transform.transform
        call={'stamp':stamp,'origin':[t.translation.x,t.translation.y],
            'array_azimuth':float(msg.azimuth_rad),'map_azimuth':wrap(yaw(t.rotation)+msg.azimuth_rad)}
        self.calls.append(call)
        self.event('name_and_audio_direction_received')

    def on_people(self,msg):
        if self.approaching:
            super().on_people(msg)
            return
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if msg.header.frame_id!='map' or not 0<=self.seconds()-stamp<=1.2:
            return
        self.visual_packet=msg
        present=set()
        for p in msg.people:
            if not p.raised_hand or p.confidence<.45:
                continue
            present.add(p.track_id)
            history=self.visual_history.setdefault(p.track_id,[])
            if history and stamp-history[-1]>.8:
                history.clear()
            if not history or stamp>history[-1]:
                history.append(stamp)
            del history[:-20]
        for track in list(self.visual_history):
            if track not in present:
                del self.visual_history[track]

    def confirm_caller(self):
        if self.visual_packet is None:
            return False
        if self.seconds()-(self.visual_packet.header.stamp.sec+self.visual_packet.header.stamp.nanosec/1e9)>1.2:
            return False
        recent=[c for c in self.calls if c['stamp']>=self.listen_since and self.seconds()-c['stamp']<20]
        if len(recent)<2 or recent[-1]['stamp']-recent[-2]['stamp']<1:
            return False
        candidates=[]
        residuals={}
        for p in self.visual_packet.people:
            history=self.visual_history.get(p.track_id,[])
            if len(history)<4 or history[-1]-history[0]<.6:
                continue
            errors=[bearing_error([p.position.x,p.position.y],c) for c in recent[-2:]]
            residuals[str(p.track_id)]=[math.degrees(e) for e in errors]
            if max(errors)<=math.radians(15):
                candidates.append(p)
        if len(candidates)!=1:
            if len(candidates)>1:
                self.data['ambiguity_events']+=1
            return False
        person=candidates[0]
        self.data.update(multimodal_confirmation=True,confirmed_track_id=person.track_id,
            confirmation_call_stamps=[c['stamp'] for c in recent[-2:]],
            candidate_audio_residuals_deg=residuals,
            candidate_count_at_confirmation=len(self.visual_packet.people))
        self.required_track=person.track_id
        self.approaching=True
        self.phase='waiting_for_raised_hand'
        self.event('unique_caller_confirmed_by_repeated_audio_and_raised_hand')
        super().on_people(self.visual_packet)
        return True

    def start_observing(self):
        self.nav_handle=None;self.nav_result=None;self.pending=None
        self.listen_since=self.seconds()
        self.observe_wall=time.monotonic()
        self.phase='observing'
        self.event('stopped_to_listen_and_observe')

    def choose_room(self):
        if self.rooms is None or not self.calls or not self.room_client.service_is_ready():
            return False
        robot=self.robot()
        call=self.calls[-1]
        choices=[]
        for room in self.rooms.rooms:
            outline=np.array([[p.x,p.y] for p in room.outline.points],dtype=np.float32)
            if room.id in self.visited or len(outline)<3 or room.area<3:
                continue
            if cv2.pointPolygonTest(outline,(float(robot[0]),float(robot[1])),False)>=0:
                continue
            centre=np.array([room.centroid.x,room.centroid.y])
            error=bearing_error(centre,call)
            if error>math.radians(80):
                continue
            choices.append((error+.025*np.linalg.norm(centre-robot),room))
        if not choices:
            return False
        _,room=min(choices,key=lambda pair:pair[0])
        self.selected_room=room
        self.room_id=room.id
        self.pending=self.room_client.call_async(GetApproachPose.Request(target=room.id))
        self.operation_deadline=time.monotonic()+20
        self.phase='resolving_room'
        self.event('request_observed_room_goal:'+room.id)
        return True

    def doorway_view(self):
        """First inspect just inside an observed door, before a room-centre goal.

        The room and door geometry are Canopy measurements of the saved SLAM
        occupancy grid. Neither scene authoring nor person truth is consulted.
        """
        if self.map is None:
            return None
        room=self.selected_room
        robot=self.robot()
        centre=np.array([room.centroid.x,room.centroid.y])
        m=self.map
        free=np.asarray(m.data).reshape(m.info.height,m.info.width)==0
        clearance=distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*m.info.resolution
        origin=np.array([m.info.origin.position.x,m.info.origin.position.y])
        angle=yaw(m.info.origin.orientation)
        rotation=np.array([[math.cos(angle),math.sin(angle)],[-math.sin(angle),math.cos(angle)]])
        doors=sorted(room.doorways,key=lambda d:np.linalg.norm(np.array([d.centre.x,d.centre.y])-robot))
        outline=np.array([[p.x,p.y] for p in room.outline.points],dtype=np.float32)
        for door in doors:
            if door.width<1.:
                continue
            d=np.array([door.centre.x,door.centre.y])
            inward=centre-d
            inward/=max(np.linalg.norm(inward),1e-6)
            for offset in (.35,.5,.7):
                point=d+inward*offset
                if len(outline)<3 or cv2.pointPolygonTest(outline,(float(point[0]),float(point[1])),False)<0:
                    continue
                col,row=np.floor(rotation@(point-origin)/m.info.resolution).astype(int)
                if (0<=row<m.info.height and 0<=col<m.info.width and clearance[row,col]>=.55):
                    return self.pose([*point,math.atan2(inward[1],inward[0])])
        return None

    def tick(self):
        # Parent owns lifecycle checks, cancellation, timeout and final visual approach.
        super().tick()
        if self.cancel_requested and self.phase in ('turn_acceptance','room_nav_acceptance'):
            if self.pending.done():
                self.nav_handle=self.pending.result()
                if self.nav_handle.accepted:
                    self.nav_result=self.nav_handle.get_result_async()
                    self.cancel_future=self.nav_handle.cancel_goal_async()
                    self.cancel_deadline=time.monotonic()+8
                    self.phase='cancelling'
                else:self.finish(False)
            return
        if self.done or self.approaching or self.cancel_requested:
            return
        now=time.monotonic()
        if self.phase=='waiting_for_name':
            if not self.calls or len(self.lifecycle_active)!=len(self.lifecycle) or not self.spin_client.server_is_ready():
                return
            call=self.calls[-1]
            if self.seconds()-call['stamp']>2:
                return
            try:
                current=self.tf.lookup_transform('map','base_footprint',Time()).transform
            except TransformException:
                return
            angle=wrap(call['map_azimuth']-yaw(current.rotation))
            self.data['requested_audio_turn_rad']=angle
            self.event('turn_to_measurement_time_audio_bearing')
            if abs(angle)<.08:
                self.start_observing()
                if self.turn_only:self.finish(True)
            else:
                # Take a partial turn, then listen again. This bounds overshoot
                # with delayed odometry without changing the frozen Nav2/gait.
                self.turn_rounds+=1
                self.pending=self.spin_client.send_goal_async(Spin.Goal(
                    target_yaw=float(.5*angle),time_allowance=Duration(seconds=45).to_msg()))
                self.phase='turn_acceptance'
        elif self.phase in ('turn_acceptance','room_nav_acceptance') and self.pending.done():
            self.nav_handle=self.pending.result()
            if not self.nav_handle.accepted:
                self.abort('Nav2 rejected voice mission action');return
            self.nav_result=self.nav_handle.get_result_async()
            self.pending=self.nav_result
            self.phase='turning' if self.phase=='turn_acceptance' else 'search_walking'
            self.event(self.phase)
        elif self.phase in ('turning','search_walking') and self.pending.done():
            previous=self.phase
            result=self.pending.result()
            if result.status!=GoalStatus.STATUS_SUCCEEDED:
                self.abort('Nav2 voice mission action failed:'+str(result.status));return
            if previous=='search_walking':
                self.visited.add(self.room_id)
                self.visual_turns=0
                self.data['visited_rooms'].append(self.room_id)
                self.event('arrived_in_observed_room:'+self.room_id)
            self.start_observing()
        elif self.phase=='observing':
            if self.turn_only:
                recent=[c for c in self.calls if c['stamp']>=self.listen_since]
                if len(recent)>=2 and self.seconds()-recent[-1]['stamp']<=2:
                    error=abs(recent[-1]['array_azimuth'])
                    self.data.update(turn_rounds=self.turn_rounds,
                        final_audio_residual_deg=math.degrees(error))
                    if error<=math.radians(15):
                        self.finish(True)
                    elif self.turn_rounds>=4:
                        self.abort('Audio turning did not converge')
                    else:self.phase='waiting_for_name'
                return
            if self.confirm_caller():
                return
            # Doorway acoustics may initially point towards a portal. Listen
            # again inside each room and orient the camera to the new bearing.
            recent=[c for c in self.calls if c['stamp']>=self.listen_since]
            if recent and now-self.observe_wall>7 and self.visual_turns<3:
                call=recent[-1]
                if self.seconds()-call['stamp']<=2 and abs(call['array_azimuth'])>math.radians(18):
                    try:
                        current=self.tf.lookup_transform('map','base_footprint',Time()).transform
                    except TransformException:
                        return
                    delta=wrap(call['map_azimuth']-yaw(current.rotation))
                    self.visual_turns+=1
                    self.pending=self.spin_client.send_goal_async(Spin.Goal(
                        target_yaw=float(.5*delta),time_allowance=Duration(seconds=45).to_msg()))
                    self.phase='turn_acceptance'
                    self.event('orient_camera_to_fresh_room_audio')
                    return
            # Wait for two independent repeated calls before resolving ambiguity.
            if now-self.observe_wall>16:
                try:
                    if not self.choose_room() and now-self.observe_wall>45:
                        self.abort('No uniquely confirmed caller or further sound-consistent observed room')
                except TransformException:
                    pass
        elif self.phase=='resolving_room':
            if now>self.operation_deadline:
                self.abort('Canopy room goal request timed out')
            elif self.pending.done():
                response=self.pending.result()
                if not response or not response.success:
                    self.visited.add(self.room_id);self.phase='observing';return
                viewpoint=self.doorway_view()
                if viewpoint is not None:
                    response.pose=viewpoint
                self.data.setdefault('search_goals',[]).append({'room':self.room_id,
                    'xy':[response.pose.pose.position.x,response.pose.pose.position.y],
                    'view':'observed doorway interior' if viewpoint else 'Canopy open spot'})
                self.pending=self.nav.send_goal_async(NavigateToPose.Goal(pose=response.pose))
                self.phase='room_nav_acceptance'
                self.event('navigate_to_observed_room:'+self.room_id)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',required=True)
    parser.add_argument('--timeout',type=float,default=600)
    parser.add_argument('--turn-only',action='store_true')
    args=parser.parse_args();rclpy.init();node=VoiceSearch(args.report,args.timeout,args.turn_only)
    try:
        while rclpy.ok() and not node.done:rclpy.spin_once(node,timeout_sec=.1)
    except KeyboardInterrupt:
        node.abort('interrupted')
        until=time.monotonic()+9
        while not node.done and rclpy.ok() and time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.1)
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    return 0 if node.data['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
