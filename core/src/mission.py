#!/usr/bin/env python3
"""Audio-guided Nav2 search, FACE-only caller confirmation, reusable per-call lifecycle."""
import json,math,time
import rclpy
from std_msgs.msg import String
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.action import ActionClient
from tf2_ros import TransformException
from nav2_msgs.action import Spin,ComputePathToPose,FollowPath,NavigateToPose
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import ClearEntireCostmap
from action_msgs.msg import GoalStatus
from rclpy.qos import QoSProfile,DurabilityPolicy
from g1_interaction_msgs.msg import CallDirection,PersonTracks
from canopy_msgs.srv import GetApproachPose
from g1_msgs.action import StepClear
from voice_search import VoiceSearch
from approach_person import Approach
from common import DATA,read_json,write_json
from caller_memory import CallerMemory
from face_approach import ranked_face_approaches
import cv2,numpy as np


class FaceVoiceSearch(VoiceSearch):
    def __init__(self):
        super().__init__(str(DATA/'mission.json'),600)
        self.identity=None;self.face_history=[];self.scan_count=0;self.reached_room_for_scan=False;self.mission_ready=False;self.pending_audio=None
        self.face_ids=list(read_json(DATA/'face-gallery.json',{}))
        self.collision_retries=0;self.collision_deadline=0.;self.clear_pending=[]
        self.costmap_stamps={};self.retry_bearing=None;self.skip_room_scan=False
        self.face_retries=0;self.face_reacquire_until=0.
        scope=self.declare_parameter('memory_scope','session-'+str(time.time_ns())).value
        self.memory=CallerMemory(DATA/'caller-memory'/(str(scope).replace('/','_')+'.json'),scope)
        self.prediction=None
        self.pending_audio_meta={};self.room_scans={};self.room_attempts={}
        self.face_pause_until=0.
        self.face_centering=False
        self.face_view_attempts=0
        self.approach_retries=0
        self.clearance_client=ActionClient(self,StepClear,'/g1_base_approach/step_clear')
        self.clearance_attempts={}
        self.clear_clients=[self.create_client(ClearEntireCostmap,'/'+name+'/clear_entirely_'+kind+'_costmap')
                            for name,kind in [('local_costmap','local'),('global_costmap','global')]]
        for kind in ('local','global'):
            self.create_subscription(Costmap,'/'+kind+'_costmap/costmap_raw',
                lambda msg,k=kind:self.costmap_stamps.__setitem__(k,msg.header.stamp.sec+msg.header.stamp.nanosec/1e9),
                QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String,'/wb/call',self.call,10)
        self.create_subscription(PersonTracks,'/wb/faces',self.on_people,10)
        self.data.update(input='propagated 4ch audio / CAMPPlus / SRP-PHAT + YuNet/SFace face-only RGB-D',identity_method='face-only, no body feature or simulator actor ID')

    def call(self,msg):
        row=json.loads(msg.data)
        # Entire active task is non-preemptive. Ignore other callers without queuing them.
        if not read_json(DATA/'ready.json',{}).get('ready') or self.done or self.cancel_requested:return
        who=row.get('identity')
        if who not in self.face_ids or row.get('keyword')!='小北小北':return
        if self.identity is not None and who!=self.identity:return
        if not math.isfinite(float(row.get('stamp',float('nan')))) or not math.isfinite(float(row.get('azimuth_rad',float('nan')))):return
        if not 0<=self.seconds()-float(row['stamp'])<=2:return
        if self.calls and float(row['stamp'])<=self.calls[-1]['stamp']:return
        if self.identity is None:
            self.identity=who;self.deadline=time.monotonic()+600;self.data['speaker_identity']=who
        direction=CallDirection(valid=True,keyword='小北',azimuth_rad=float(row['azimuth_rad']))
        direction.header.frame_id='microphone_array';direction.header.stamp=Time(seconds=float(row['stamp'])).to_msg()
        # Physical camera/audio can arrive just before their matching dynamic TF.
        # Retry the measurement-time transform, never substitute today's transform.
        self.pending_audio=direction
        self.pending_audio_meta={'half_angle_deg':row.get('half_angle_deg',30.),
            'window_spread_deg':row.get('window_spread_deg'),
            'angular_candidates':row.get('angular_candidates',[]),
            'frequency_candidates':row.get('frequency_candidates',[])}
        self.publish_state()

    def on_call(self,msg):
        # Baseline topic is intentionally inactive in workbench mode.
        pass

    def on_people(self,msg):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if msg.header.frame_id!='map' or not 0<=self.seconds()-stamp<=1.2:return
        if read_json(DATA/'ready.json',{}).get('ready'):
            for person in msg.people:
                if person.confidence>=.45 and 1<=person.track_id<=len(self.face_ids):
                    self.memory.observe_face(self.face_ids[person.track_id-1],
                        [person.position.x,person.position.y],stamp)
        if self.identity is None or self.identity not in self.face_ids:return
        required=self.face_ids.index(self.identity)+1
        packet=PersonTracks();packet.header=msg.header
        for person in msg.people:
            if person.track_id==required:
                # Adapter flag enables the existing safe-approach geometry/controller only.
                # It is NOT a raised-hand measurement; external /wb/faces retains False.
                person.raised_hand=True;packet.people.append(person)
        if self.approaching:
            previous_lock=self.locked
            Approach.on_people(self,packet)
            if previous_lock is None and self.locked is not None and self.phase=='planning':
                self.prefer_visible_face_approach()
            self.accept_already_beside();return
        self.visual_packet=packet
        if not packet.people:self.face_history=[];return
        if self.face_history and stamp-self.face_history[-1]>.9:self.face_history=[]
        self.face_history.append(stamp);self.face_history=self.face_history[-10:]

    def confirm_caller(self):
        if not self.face_is_confirmable():return False
        person=self.visual_packet.people[0]
        self.data.update(multimodal_confirmation=True,confirmed_face_identity=self.identity,confirmation_source='CAMPPlus voice gallery -> SFace face gallery; no gesture or body identity',confirmed_track_id=person.track_id)
        self.required_track=person.track_id;self.approaching=True;self.phase='waiting_for_raised_hand'
        if self.face_reacquire_until:
            self.event('same_caller_face_reacquired_after_safe_stop')
            self.face_reacquire_until=0.
        self.event('speaker_and_face_identity_confirmed')
        # Preserve original temporal confidence and safe approach planner after confirmation.
        Approach.on_people(self,self.visual_packet)
        return True

    def face_is_confirmable(self):
        return bool(self.visual_packet and self.visual_packet.people and len(self.face_history)>=3
            and self.face_history[-1]-self.face_history[0]>=.4 and self.seconds()-self.face_history[-1]<=1.2)

    def face_candidate_visible(self):
        # One matching face is sufficient to PAUSE looking, never to approach.
        # Otherwise a face can cross the camera before three frames accumulate.
        if not self.visual_packet or not self.visual_packet.people:return False
        h=self.visual_packet.header.stamp;stamp=h.sec+h.nanosec/1e9
        return 0<=self.seconds()-stamp<=.7

    def event(self,text):
        if text=='gesture_confirmed':text='face_temporal_confirmation_complete'
        super().event(text);self.publish_state()

    def publish_state(self):
        write_json(DATA/'mission-state.json',{'identity':self.identity,'phase':self.phase,'busy':self.identity is not None and not self.done,'done':self.done,'passed':self.data.get('passed',False),'failure':self.data.get('failure'),'events':self.data.get('events',[])[-8:],'prediction':self.prediction})

    def accept_already_beside(self):
        # Reuse the frozen 1.0--1.95 m acceptance and three-second stop check.
        # A caller already beside us needs no artificial approach loop.
        if self.phase!='planning' or self.locked is None or self.target is None:return False
        if self.nav_result is not None and not self.nav_result.done():return False
        try:distance=float(np.linalg.norm(self.robot()-self.target))
        except TransformException:return False
        if not 1.0<=distance<=1.95:return False
        self.phase='checking_stop';self.pending=None;self.settle_since=None
        self.data['arrival_mode']='already within accepted person standoff; verify fresh face and 3-second stationary odometry'
        self.event('already_beside_confirmed_person_check_stop');return True

    def prefer_visible_face_approach(self):
        # Faces actually seen by RGB-D, never scene actor positions. This only
        # ranks the frozen safe endpoint generator, without weakening clearance.
        others=[]
        for identity in self.face_ids:
            if identity==self.identity:continue
            prior=self.memory.visual_prior(identity)
            if prior and prior['age_s']<=15:others.append(prior['position'])
        m=self.map;q=m.info.origin.orientation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        try:robot=self.robot()
        except TransformException:return
        candidates=ranked_face_approaches(self.target,robot,np.asarray(m.data).reshape(m.info.height,m.info.width),
            m.info.resolution,[m.info.origin.position.x,m.info.origin.position.y,yaw],others,max_standoff=1.65)
        # Nav2 permits 0.25 m endpoint tolerance. The old 1.85 m outer
        # candidate could therefore finish outside the unchanged 1.95 m
        # person-distance acceptance. Search viewpoints may still use 1.85 m;
        # final approach endpoints leave room for controller tolerance.
        self.candidates=iter(candidates)
        self.data.setdefault('face_view_preferences',[]).append({'stamp':self.seconds(),
            'source':'recent RGB-D face measurements only','other_face_count':len(others),
            'preferred_endpoints':[list(p) for p in candidates[:3]]})

    def start_face_viewpoint(self,max_age_s=15.):
        # Re-observation is a search, not a blind final approach. The preceding
        # visual approach has already been cancelled and acknowledged stopped.
        # Only a recent RGB-D face measurement supplies the bounded search hint.
        if self.face_view_attempts>=2 or time.monotonic()-self.face_observed_wall>max_age_s:return False
        if not self.nav.server_is_ready() or self.map is None:return False
        try:robot=self.robot()
        except TransformException:return False
        m=self.map;q=m.info.origin.orientation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        others=[]
        for identity in self.face_ids:
            prior=self.memory.visual_prior(identity)
            if identity!=self.identity and prior and prior['age_s']<=15:others.append(prior['position'])
        candidates=ranked_face_approaches(self.face_last_position,robot,
            np.asarray(m.data).reshape(m.info.height,m.info.width),m.info.resolution,
            [m.info.origin.position.x,m.info.origin.position.y,yaw],others)
        for goal in candidates:
            travel=float(np.linalg.norm(np.asarray(goal[:2])-robot))
            standoff=float(np.linalg.norm(np.asarray(goal[:2])-self.face_last_position))
            if not .35<=travel<=4. or standoff<1.59:continue
            self.face_view_attempts+=1
            self.data.setdefault('face_view_searches',[]).append({'attempt':self.face_view_attempts,
                'stamp':self.seconds(),'goal':list(goal),'last_measured_face':self.face_last_position.tolist(),
                'face_age_wall_s':time.monotonic()-self.face_observed_wall,
                'max_hint_age_s':max_age_s,
                'hint_distance_m':travel,'time_limit_wall_s':18.,
                'source':'last RGB-D face + observed map; fresh face required for final approach'})
            self.face_view_deadline=time.monotonic()+18
            self.pending=self.nav.send_goal_async(NavigateToPose.Goal(pose=self.pose(goal)))
            self.phase='face_view_acceptance';self.event('bounded_nav2_face_reobservation_search');return True
        return False

    def tick_face_viewpoint(self):
        Approach.tick(self)
        if self.done:return
        if self.phase=='face_view_acceptance' and self.pending.done():
            self.nav_handle=self.pending.result()
            if not self.nav_handle.accepted:
                if self.cancel_requested:self.finish(False)
                else:self.abort('Nav2 rejected face reobservation viewpoint')
                return
            self.nav_result=self.nav_handle.get_result_async();self.pending=self.nav_result
            if self.cancel_requested:
                self.cancel_future=self.nav_handle.cancel_goal_async();self.cancel_deadline=time.monotonic()+8
                self.phase='cancelling';return
            self.phase='face_view_walking';self.event('walking_to_face_reobservation_viewpoint')
        if self.phase!='face_view_walking' or self.cancel_requested:return
        if self.nav_result.done():
            result=self.nav_result.result()
            self.data['face_view_searches'][-1]['navigation_terminal_status']=int(result.status)
            if result.status!=GoalStatus.STATUS_SUCCEEDED:
                self.abort('Nav2 face reobservation viewpoint failed');return
            self.face_reacquire_until=max(self.face_reacquire_until,time.monotonic()+4.)
            self.start_observing();self.event('face_reobservation_viewpoint_reached')
        elif self.face_is_confirmable():
            self.cancel_future=self.nav_handle.cancel_goal_async()
            self.scan_cancel_deadline=time.monotonic()+8;self.phase='face_scan_stop'
            self.event('caller_face_reconfirmed_stop_viewpoint_search')
        elif time.monotonic()>self.face_view_deadline:
            self.abort('bounded face viewpoint search timed out; staying stopped')

    def start_observing(self):
        # A rejected Canopy goal also enters `visited`; it is not a reached room.
        # Scan only after an actual successful room navigation, never while a
        # room service request is resolving or after an unreachable-room reply.
        if self.phase=='search_walking':
            self.reached_room_for_scan=True;self.scan_count=0;self.skip_room_scan=False
        self.face_centering=False
        super().start_observing()
        self.face_history=[]

    def choose_room(self):
        # Reacquire the already-confirmed caller while stopped; do not start
        # another room route on an old acoustic bearing during this window.
        if self.face_reacquire_until:return False
        if self.identity and self.try_nearby_memory_view():return True
        if self.reached_room_for_scan and self.scan_count>=3 and self.room_id not in self.room_scans:
            self.room_scans[self.room_id]={'count':1,'wall':time.time(),'stamp':self.seconds(),
                'evidence':'completed limited camera scan without confirmed caller; not whole-room clearance'}
        prediction=self.refresh_prediction()
        # A recent face observation is only a priority hint for Canopy's
        # observed room goals. Nav2 still checks every path; final arrival
        # still requires fresh same-identity face confirmation.
        if prediction and self.rooms and self.room_client.service_is_ready():
            robot=self.robot()
            choices=[]
            failed={r['room'] for r in self.data.get('unreachable_rooms',[])}
            for candidate in prediction['room_candidates']:
                if candidate['area_m2']<3 or candidate['id'] in failed:continue
                room=next(r for r in self.rooms.rooms if r.id==candidate['id'])
                outline=np.asarray([[p.x,p.y] for p in room.outline.points],np.float32)
                if len(outline)<3 or cv2.pointPolygonTest(outline,tuple(map(float,robot)),False)>=0:continue
                attempts=self.room_attempts.get(room.id,0)
                # Inspect unvisited hypotheses first. At most one revisit,
                # supported by a fresh call after this room's actual scan.
                if room.id in self.visited:
                    scan=self.room_scans.get(room.id)
                    if not scan or self.calls[-1]['stamp']<=scan['stamp'] or attempts>=2:continue
                choices.append((room.id not in self.visited,candidate['priority'],room))
            if choices:
                _,_,room=max(choices,key=lambda row:row[:2])
                self.selected_room=room;self.room_id=room.id
                self.room_attempts[room.id]=self.room_attempts.get(room.id,0)+1
                self.pending=self.room_client.call_async(GetApproachPose.Request(target=room.id))
                self.operation_deadline=time.monotonic()+20;self.phase='resolving_room'
                self.reached_room_for_scan=False
                self.data.setdefault('search_decisions',[]).append({'room':room.id,'prediction':prediction})
                self.event('request_observed_room_goal:'+room.id);return True
        return False

    def try_nearby_memory_view(self):
        # Do not exclude the robot's current room when the caller was recently
        # seen nearby but is now occluded. One local re-observation is cheaper
        # than leaving that room merely because its label is "current".
        if self.data.get('remembered_view_search'):return False
        prior=self.memory.visual_prior(self.identity)
        if not prior or prior['age_s']>60:return False
        try:robot=self.robot()
        except TransformException:return False
        point=np.asarray(prior['position'],float)
        if float(np.linalg.norm(point-robot))>5.5:return False
        self.face_last_position=point;self.face_observed_wall=time.monotonic()-prior['age_s']
        if not self.start_face_viewpoint(max_age_s=60.):return False
        self.face_reacquire_until=time.monotonic()+24.
        self.data['remembered_view_search']={'age_s':prior['age_s'],'position':point.tolist(),
            'source':prior['source'],'final_acceptance_requires_fresh_face':True}
        self.event('inspect_nearby_remembered_face_before_leaving_room');return True

    def refresh_prediction(self):
        if not self.calls or not self.identity:return None
        rooms=[]
        if self.rooms:
            rooms=[{'id':r.id,'centroid':[r.centroid.x,r.centroid.y],'area':r.area,
                'outline':[[p.x,p.y] for p in r.outline.points],
                'doorways':[{'centre':[d.centre.x,d.centre.y],'to_room':d.to_room} for d in r.doorways]} for r in self.rooms.rooms]
        try:robot=self.robot()
        except TransformException:robot=None
        self.prediction=self.memory.predict(self.identity,self.calls[-1],rooms,scans=self.room_scans,robot=robot)
        self.data['position_prediction']=self.prediction
        return self.prediction

    def doorway_view(self):
        # Retain Canopy's reachable, maximum-clearance room viewpoint. The old
        # shortcut 35 cm inside a noisy segmented doorway repeatedly stalled at
        # wall corners even though the room's central free space was reachable.
        return None

    def abort(self,reason):
        # Even a user interruption during goal acceptance must await the handle
        # and cancel it; returning early could otherwise leave Nav2 moving.
        if self.phase in ('face_view_acceptance','clearance_acceptance') and not self.cancel_requested and not self.done:
            self.data['failure']=reason;self.cancel_requested=True;self.event(reason);return
        if reason.startswith('Nav2 approach failed:') and self.recover_approach_terminal():return
        if reason.startswith('Nav2 voice mission action failed:'):
            try:
                result=self.pending.result().result
                code=int(getattr(result,'error_code',0))
                text=str(getattr(result,'error_msg',''))
                collision=code==getattr(Spin.Result,'COLLISION_AHEAD',-1)
                self.data['navigation_failure']={'error_code':code,'error_msg':text,'phase':self.phase}
                if collision and self.phase=='turning' and not self.cancel_requested:
                    if self.recover_turn_collision():return
                recoverable=(ComputePathToPose.Result.NO_VALID_PATH,ComputePathToPose.Result.TIMEOUT,
                    ComputePathToPose.Result.GOAL_OCCUPIED,FollowPath.Result.CONTROLLER_TIMED_OUT,
                    FollowPath.Result.PATIENCE_EXCEEDED,FollowPath.Result.FAILED_TO_MAKE_PROGRESS)
                if self.phase=='search_walking' and not self.cancel_requested and code in recoverable:
                    if self.start_clearance_recovery(code,text):return
                    # Planner NO_VALID_PATH / timeout or controller patience.
                    # The action is already terminal: try another observed
                    # Canopy room, bounded by the finite visited-room set.
                    self.data.setdefault('unreachable_rooms',[]).append({'room':self.room_id,
                        'error_code':code,'error_msg':text,'sim_time':self.seconds()})
                    self.visited.add(self.room_id)
                    self.reached_room_for_scan=False;self.skip_room_scan=True
                    self.phase='observing'  # A failed route is not a room arrival.
                    self.start_observing();self.observe_wall=time.monotonic()-16
                    self.event('room_route_failed_try_other_observed_room')
                    return
                self.data['failure_message']=('已识别呼叫；Nav2 检测到转向碰撞风险，已安全停止。请检查周围障碍和定位。' if collision else '已识别呼叫；Nav2 动作中止，已停止。请检查周围障碍、定位与导航日志。')
            except (AttributeError,TypeError):pass
        if (reason=='visual track lost or stale; cancel navigation' and
                self.phase in ('walking','checking_stop') and not self.cancel_requested and self.face_retries<2):
            # Stop first. A brief loss at a navigation heading change is not
            # evidence of a different person, and never permits blind walking.
            self.face_retries+=1;self.face_last_position=self.target.copy()
            self.face_observed_wall=self.last_seen
            self.data.setdefault('face_recovery',[]).append({'attempt':self.face_retries,'sim_time':self.seconds()})
            if self.nav_handle and self.nav_handle.accepted:
                self.cancel_future=self.nav_handle.cancel_goal_async()
                self.face_cancel_deadline=time.monotonic()+8
                self.phase='face_recovery_stop';self.event('face_lost_cancel_before_reacquisition')
                return
        super().abort(reason)

    def recover_approach_terminal(self):
        # Only a completed, aborted navigation action can release motion
        # ownership. A rejected/cancelled/pending action is never an arrival.
        if (self.phase!='walking' or self.cancel_requested or self.done or
                self.nav_result is None or not self.nav_result.done()):return False
        result=self.nav_result.result()
        code=int(getattr(result.result,'error_code',0))
        recoverable=(ComputePathToPose.Result.NO_VALID_PATH,ComputePathToPose.Result.TIMEOUT,
            ComputePathToPose.Result.GOAL_OCCUPIED,FollowPath.Result.CONTROLLER_TIMED_OUT,
            FollowPath.Result.PATIENCE_EXCEEDED,FollowPath.Result.FAILED_TO_MAKE_PROGRESS)
        if result.status!=GoalStatus.STATUS_ABORTED or code not in recoverable:return False
        row={'sim_time':self.seconds(),'status':int(result.status),'error_code':code,
             'error_msg':str(getattr(result.result,'error_msg','')),
             'failed_endpoint':list(self.goal)}
        self.data.setdefault('approach_terminal_recovery',[]).append(row)
        # Require a current measurement of the same locked face and a fresh
        # localization before applying the existing distance/stop acceptance.
        fresh_face=(self.people is not None and self.locked is not None and
            time.monotonic()-self.last_seen<=1. and
            0<=self.seconds()-(self.people.header.stamp.sec+self.people.header.stamp.nanosec/1e9)<=1.2 and
            any(p.track_id==self.locked and p.confidence>=.45 for p in self.people.people))
        try:
            t=self.tf.lookup_transform('map','base_footprint',Time())
            stamp=t.header.stamp.sec+t.header.stamp.nanosec/1e9
            if not 0<=self.seconds()-stamp<=1.:return False
        except TransformException:return False
        self.nav_handle=None;self.nav_result=None;self.pending=None
        self.phase='planning'
        if fresh_face and self.accept_already_beside():
            row['outcome']='within existing standoff; require continuing fresh face and 3-second stationary odometry'
            self.event('nav2_terminal_failure_but_person_standoff_reached_verify_stop');return True
        # The iterator contains only the original collision-checked endpoints;
        # never retry the failed endpoint or weaken the footprint/clearance.
        if fresh_face and self.approach_retries<2:
            self.approach_retries+=1;row['outcome']='try next original safe endpoint'
            self.event('nav2_approach_terminal_failure_try_next_safe_endpoint');return True
        self.phase='walking'
        row['outcome']='no fresh face or retry budget exhausted; fail safely'
        return False

    def start_clearance_recovery(self,code,text):
        # The failed Nav2 action is terminal before StepClear takes ownership.
        # Never infer that every route/room is blocked from one stuck origin.
        if self.clearance_attempts.get(self.room_id,0)>=1 or sum(self.clearance_attempts.values())>=2:
            return False
        if not self.clearance_client.server_is_ready():return False
        self.clearance_attempts[self.room_id]=self.clearance_attempts.get(self.room_id,0)+1
        self.data.setdefault('clearance_recovery',[]).append({'room':self.room_id,
            'sim_time':self.seconds(),'trigger_error_code':code,'trigger_error_msg':text,
            'requested_clearance_m':.7,'timeout_s':20.,
            'controller':'upstream Grove StepClear -> velocity smoother -> collision monitor'})
        self.nav_handle=None;self.nav_result=None
        self.pending=self.clearance_client.send_goal_async(StepClear.Goal(clearance_m=.7,timeout_s=20.))
        self.phase='clearance_acceptance';self.clearance_deadline=time.monotonic()+30.
        self.event('nav2_terminal_failure_try_observed_safe_clearance');return True

    def tick_clearance_recovery(self):
        Approach.tick(self)
        if self.done:return
        if self.phase=='clearance_acceptance' and self.pending.done():
            self.nav_handle=self.pending.result()
            if not self.nav_handle.accepted:
                if self.cancel_requested:self.finish(False);return
                self.phase='clearance_rejected';self.abort('Safe clearance action rejected; staying stopped');return
            self.nav_result=self.nav_handle.get_result_async();self.pending=self.nav_result
            if self.cancel_requested:
                self.cancel_future=self.nav_handle.cancel_goal_async();self.cancel_deadline=time.monotonic()+8
                self.phase='cancelling';return
            self.phase='clearance_walking';self.event('grove_safe_clearance_running')
        if self.phase!='clearance_walking' or self.cancel_requested:return
        if self.nav_result.done():
            response=self.nav_result.result();result=response.result
            self.data['clearance_recovery'][-1].update(status=int(response.status),
                success=bool(result.success),clearance_m=float(result.clearance_m),message=result.message)
            self.nav_handle=None;self.nav_result=None;self.pending=None
            if response.status!=GoalStatus.STATUS_SUCCEEDED or not result.success:
                self.phase='clearance_failed'
                self.abort('Safe clearance could not complete; blocked at current pose, room not declared inspected');return
            self.pending=self.room_client.call_async(GetApproachPose.Request(target=self.room_id))
            self.operation_deadline=time.monotonic()+20;self.phase='resolving_room'
            self.event('safe_clearance_complete_retry_same_observed_room')
        elif time.monotonic()>self.clearance_deadline:
            self.abort('Safe clearance result timed out; cancel before stopping')

    def tick_face_recovery(self):
        if self.cancel_requested or self.done:return
        if self.nav_result is None or not self.nav_result.done():
            if time.monotonic()>self.face_cancel_deadline:
                self.abort('face reacquisition: navigation cancellation not acknowledged')
            return
        result=self.nav_result.result()
        if result.status not in (GoalStatus.STATUS_CANCELED,GoalStatus.STATUS_SUCCEEDED,GoalStatus.STATUS_ABORTED):
            self.abort('face reacquisition: navigation has not stopped');return
        self.data['face_recovery'][-1]['navigation_terminal_status']=int(result.status)
        self.nav_handle=None;self.nav_result=None;self.pending=None
        self.locked=None;self.target=None;self.confirm_id=None;self.confirm_stamps=[]
        self.approaching=False;self.face_history=[];self.visual_packet=None
        self.reached_room_for_scan=False;self.skip_room_scan=True
        self.face_reacquire_until=time.monotonic()+20
        self.start_observing()
        try:
            t=self.tf.lookup_transform('map','base_footprint',Time())
            stamp=t.header.stamp.sec+t.header.stamp.nanosec/1e9
            if not 0<=self.seconds()-stamp<1.:raise ValueError('stale localization')
            q=t.transform.rotation
            heading=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            p=t.transform.translation;delta=self.face_last_position-[p.x,p.y]
            bearing=math.atan2(delta[1],delta[0])
            angle=math.atan2(math.sin(bearing-heading),math.cos(bearing-heading))
        except (TransformException,ValueError):
            self.abort('face reacquisition: fresh localization unavailable');return
        if abs(angle)>.08:
            self.face_centering=True
            self.turn_target_bearing=heading+angle
            self.pending=self.spin_client.send_goal_async(Spin.Goal(target_yaw=float(angle),time_allowance=Duration(seconds=15).to_msg()))
            self.phase='turn_acceptance';self.event('orient_to_last_camera_observed_face')

    def recover_turn_collision(self):
        """Refresh live Nav2 maps once, then use the existing room planner.

        No direct velocity, simulator pose, obstacle erasure or motion outside
        Nav2. The accepted failed Spin has already returned its terminal result.
        """
        self.nav_handle=None;self.nav_result=None;self.pending=None
        self.data.setdefault('collision_recovery',[]).append({'attempt':self.collision_retries+1,
            'phase':self.phase,'sim_time':self.seconds(),'error_code':703})
        if self.collision_retries>=1:
            self.skip_room_scan=True;self.reached_room_for_scan=False
            self.start_observing()
            self.event('turn_blocked_choose_observed_room_route')
            if self.confirm_caller():return True
            try:selected=self.choose_room()
            except TransformException:selected=False
            if not selected:
                # Stay stopped while Canopy/vision updates; do not repeat Spin.
                self.observe_wall=time.monotonic()-16
            return True
        self.collision_retries+=1
        try:
            transform=self.tf.lookup_transform('map','base_footprint',Time()).transform
            q=transform.rotation
            current=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            # Resume the same heading, including an interrupted room scan.
            self.retry_bearing=getattr(self,'turn_target_bearing',current)
        except TransformException:
            self.retry_bearing=None
        self.clear_pending=[]
        for client in self.clear_clients:
            if client.service_is_ready():self.clear_pending.append(client.call_async(ClearEntireCostmap.Request()))
        self.clear_sim_time=self.seconds();self.collision_deadline=time.monotonic()+8
        self.collision_wait_until=time.monotonic()+1.2
        self.phase='collision_refresh';self.event('collision_refresh_live_costmaps')
        return True

    def tick_collision_recovery(self):
        if self.cancel_requested or self.done:return
        now=time.monotonic()
        fresh=all(self.costmap_stamps.get(k,-1)>self.clear_sim_time+.4
                  and 0<=self.seconds()-self.costmap_stamps.get(k,-1)<1.5 for k in ('local','global'))
        if now<self.collision_wait_until:return
        if not fresh or any(not f.done() for f in self.clear_pending):
            if now>self.collision_deadline:self.abort('collision recovery: live costmaps did not refresh; staying stopped')
            return
        if any(f.exception() is not None for f in self.clear_pending):
            self.abort('collision recovery: costmap service failed; staying stopped');return
        try:
            t=self.tf.lookup_transform('map','base_footprint',Time())
            stamp=t.header.stamp.sec+t.header.stamp.nanosec/1e9
            if not 0<=self.seconds()-stamp<1.:raise ValueError('stale localization')
            q=t.transform.rotation;heading=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        except (TransformException,ValueError):
            if now>self.collision_deadline:self.abort('collision recovery: localization unavailable; staying stopped')
            return
        if self.retry_bearing is None:
            self.start_observing();return
        angle=math.atan2(math.sin(self.retry_bearing-heading),math.cos(self.retry_bearing-heading))
        if abs(angle)<.08:self.start_observing();return
        self.pending=self.spin_client.send_goal_async(Spin.Goal(target_yaw=float(angle),time_allowance=Duration(seconds=35).to_msg()))
        self.phase='turn_acceptance';self.event('retry_turn_after_fresh_observations')

    def tick(self):
        if (DATA/'cancel-request').exists():
            (DATA/'cancel-request').unlink()
            if self.identity:self.abort('user interrupted; safe stop')
        if self.identity is None:
            self.deadline=time.monotonic()+600
        if self.phase in ('clearance_acceptance','clearance_walking'):
            self.tick_clearance_recovery();self.publish_state();return
        if self.phase=='face_recovery_stop':
            self.tick_face_recovery();self.publish_state();return
        if self.phase in ('face_view_acceptance','face_view_walking'):
            self.tick_face_viewpoint();self.publish_state();return
        if self.phase=='face_scan_stop':
            Approach.tick(self)
            if not self.cancel_requested and not self.done:
                if self.nav_result.done():
                    if self.nav_result.result().status not in (GoalStatus.STATUS_SUCCEEDED,GoalStatus.STATUS_CANCELED):
                        self.abort('scan action failed while stopping for a face');return
                    # A single face can be at the image edge. After the action
                    # has stopped, centre on that measured face before waiting
                    # for temporal confirmation; never chase simulator poses.
                    point=None
                    if self.face_candidate_visible():
                        p=self.visual_packet.people[0].position;point=np.array([p.x,p.y])
                    self.start_observing();self.face_pause_until=time.monotonic()+2.
                    if point is not None:
                        try:
                            t=self.tf.lookup_transform('map','base_footprint',Time())
                            stamp=t.header.stamp.sec+t.header.stamp.nanosec/1e9
                            if not 0<=self.seconds()-stamp<1.:raise ValueError('stale localization')
                            q=t.transform.rotation
                            heading=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
                            p=t.transform.translation;delta=point-[p.x,p.y]
                            angle=math.atan2(math.sin(math.atan2(delta[1],delta[0])-heading),math.cos(math.atan2(delta[1],delta[0])-heading))
                            if abs(angle)>.12:
                                self.face_centering=True;self.turn_target_bearing=heading+angle
                                self.pending=self.spin_client.send_goal_async(Spin.Goal(target_yaw=float(angle),time_allowance=Duration(seconds=15).to_msg()))
                                self.phase='turn_acceptance';self.event('centre_on_camera_measured_face_before_confirmation')
                        except (TransformException,ValueError):pass
                    if self.phase=='observing':self.confirm_caller()
                elif time.monotonic()>self.scan_cancel_deadline:self.abort('face scan cancellation not acknowledged; staying stopped')
            self.publish_state();return
        if self.face_reacquire_until and time.monotonic()>self.face_reacquire_until and not self.approaching:
            self.abort('face reacquisition timed out; staying stopped')
        if self.pending_audio is not None:
            stamp=self.pending_audio.header.stamp.sec+self.pending_audio.header.stamp.nanosec/1e9
            if not 0<=self.seconds()-stamp<=2:
                self.pending_audio=None
                if not self.calls:self.abort('Audio measurement expired before matching TF became available')
                else:self.event('expired_repeat_call_ignored_without_interrupting_task')
            else:
                before=len(self.calls)
                super().on_call(self.pending_audio)
                if len(self.calls)>before:
                    self.calls[-1].update(self.pending_audio_meta)
                    call=self.calls[-1];rotation=call['map_azimuth']-call['array_azimuth']
                    call['map_hypotheses']=[{'map_azimuth':call['map_azimuth'],'weight':.6,'band':'full'}]
                    for candidate in call.get('frequency_candidates',[]):
                        if candidate.get('peak_contrast',0)<.08:continue
                        angle=float(candidate['azimuth_rad'])+rotation
                        call['map_hypotheses'].append({'map_azimuth':math.atan2(math.sin(angle),math.cos(angle)),
                            'weight':.2,'band':candidate['band_hz']})
                    self.memory.observe_audio(self.identity,self.calls[-1]);self.refresh_prediction()
                    self.pending_audio=None
        # Record the requested heading from measured map TF before any Spin is
        # sent. A retry uses remaining angle, never another blind full rotation.
        previous=self.phase;heading=None
        if previous=='collision_refresh':
            Approach.tick(self);self.tick_collision_recovery()
        else:
            try:
                t=self.tf.lookup_transform('map','base_footprint',Time()).transform
                q=t.rotation;heading=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            except TransformException:heading=None
            if (previous in ('turning','search_walking') and not self.face_centering and not self.cancel_requested and self.nav_handle
                    and not self.nav_result.done() and self.face_candidate_visible()):
                self.cancel_future=self.nav_handle.cancel_goal_async()
                self.scan_cancel_deadline=time.monotonic()+8;self.phase='face_scan_stop'
                self.event('same_face_seen_stop_scan_before_approach');self.publish_state();return
            if previous=='observing' and self.identity and not self.approaching:
                Approach.tick(self)
                if not self.cancel_requested and not self.done:self.tick_search_observation(heading)
            elif (previous=='waiting_for_name' and self.calls and heading is not None
                  and not self.cancel_requested and len(self.lifecycle_active)==len(self.lifecycle)
                  and self.spin_client.server_is_ready() and self.seconds()-self.calls[-1]['stamp']<=2):
                Approach.tick(self)
                if not self.cancel_requested and not self.done:
                    delta=math.atan2(math.sin(self.calls[-1]['map_azimuth']-heading),math.cos(self.calls[-1]['map_azimuth']-heading))
                    self.turn_target_bearing=heading+delta
                    if abs(delta)<.08:self.start_observing()
                    else:
                        self.pending=self.spin_client.send_goal_async(Spin.Goal(target_yaw=float(delta),time_allowance=Duration(seconds=35).to_msg()))
                        self.phase='turn_acceptance';self.event('direct_safe_turn_to_measured_audio')
            else:super().tick()
            if self.phase=='turn_acceptance' and previous!='turn_acceptance' and heading is not None:
                if self.calls and previous not in ('waiting_for_name','observing'):
                    delta=math.atan2(math.sin(self.calls[-1]['map_azimuth']-heading),math.cos(self.calls[-1]['map_azimuth']-heading))
                    self.turn_target_bearing=heading+.5*delta
        self.publish_state()

    def tick_search_observation(self,heading):
        if self.confirm_caller():return
        dwell=time.monotonic()-self.observe_wall
        if self.face_reacquire_until:
            if dwell>=2.:self.start_face_viewpoint()
            return
        if time.monotonic()<self.face_pause_until:return
        if dwell<.8 or heading is None:return
        if not self.skip_room_scan and self.scan_count<3:
            # Three 120-degree segments cover the rear as well as the front.
            # Fresh matching faces interrupt the scan via an acknowledged stop.
            self.scan_count+=1;angle=2*math.pi/3
            self.turn_target_bearing=heading+angle
            self.pending=self.spin_client.send_goal_async(Spin.Goal(target_yaw=angle,time_allowance=Duration(seconds=35).to_msg()))
            self.phase='turn_acceptance';self.event('full_surround_face_scan_segment');return
        if not self.choose_room() and dwell>12:
            self.abort('Search routes exhausted without fresh caller-face confirmation; failed routes were not inspected')


def main():
    rclpy.init();node=FaceVoiceSearch()
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=.1)
            if node.done:
                # Parent has waited for navigation completion/cancellation. No automatic replay.
                write_json(DATA/'last-mission.json',node.data)
                time.sleep(.3);node.destroy_node();node=FaceVoiceSearch()
    except KeyboardInterrupt:
        if node.identity:
            node.abort('shutdown')
            end=time.monotonic()+8
            while rclpy.ok() and not node.done and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
