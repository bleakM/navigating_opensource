"""Isolated scheduler/error reporting regression; no ROS spin or motion calls."""
import sys,tempfile,time,unittest
from concurrent.futures import Future
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mission import FaceVoiceSearch
from voice_search import VoiceSearch
from approach_person import Approach
from nav2_msgs.action import Spin,ComputePathToPose
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import TransformStamped


class MissionTests(unittest.TestCase):
    def test_unreachable_room_does_not_trigger_scan_or_overwrite_pending_request(self):
        node=FaceVoiceSearch.__new__(FaceVoiceSearch)
        node.identity='caller';node.done=False;node.approaching=False;node.pending_audio=None
        node.reached_room_for_scan=False;node.visited={'unreachable'};node.observe_wall=time.monotonic()-100;node.scan_count=0
        node.phase='observing';node.skip_room_scan=False;node.calls=[]
        node.face_reacquire_until=0.;node.cancel_requested=False;node.approaching=False
        node.face_centering=False
        node.tick_search_observation=Mock()
        node.tf=SimpleNamespace(lookup_transform=lambda *args:TransformStamped())
        node.publish_state=lambda:None;node.confirm_caller=lambda:False
        sent=[];node.spin_client=SimpleNamespace(send_goal_async=sent.append)
        with tempfile.TemporaryDirectory() as folder,patch('mission.DATA',Path(folder)),patch.object(Approach,'tick'):
            # Parent may return a failed room reply to observing. It is not arrival.
            with patch.object(VoiceSearch,'tick',lambda self:setattr(self,'phase','observing')):node.tick()
            self.assertFalse(sent)
            # Parent chooses a room: a pending service future must not be replaced by Spin.
            pending=object();node.pending=pending;node.reached_room_for_scan=True
            with patch.object(VoiceSearch,'tick',lambda self:setattr(self,'phase','resolving_room')):node.tick()
            self.assertIs(node.pending,pending);self.assertFalse(sent)

    def test_surround_scan_covers_rear_without_long_fixed_wait(self):
        import math
        node=FaceVoiceSearch.__new__(FaceVoiceSearch)
        node.confirm_caller=Mock(return_value=False);node.face_reacquire_until=0.
        node.observe_wall=time.monotonic()-1.;node.skip_room_scan=False;node.scan_count=0
        node.face_pause_until=0.
        node.spin_client=Mock();node.event=Mock();node.choose_room=Mock(return_value=True)
        for _ in range(3):node.tick_search_observation(0.)
        angles=[c.args[0].target_yaw for c in node.spin_client.send_goal_async.call_args_list]
        self.assertAlmostEqual(sum(angles),2*math.pi,places=5)
        node.tick_search_observation(0.);node.choose_room.assert_called_once()

    def test_face_seen_during_scan_requires_terminal_cancel_before_approach(self):
        node=FaceVoiceSearch.__new__(FaceVoiceSearch)
        node.phase='face_scan_stop';node.identity='caller';node.done=False;node.cancel_requested=False
        node.nav_result=Future();node.scan_cancel_deadline=time.monotonic()+8
        node.publish_state=Mock();node.start_observing=Mock(side_effect=lambda:setattr(node,'phase','observing'));node.confirm_caller=Mock()
        node.face_candidate_visible=Mock(return_value=False)
        with tempfile.TemporaryDirectory() as folder,patch('mission.DATA',Path(folder)),patch.object(Approach,'tick'):
            node.tick();node.confirm_caller.assert_not_called()
            node.nav_result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
            node.tick();node.start_observing.assert_called_once();node.confirm_caller.assert_called_once()

    def test_edge_face_is_centred_after_scan_cancel_before_confirmation(self):
        import math
        node=FaceVoiceSearch.__new__(FaceVoiceSearch)
        node.phase='face_scan_stop';node.identity='caller';node.done=False;node.cancel_requested=False
        node.nav_result=Future();node.nav_result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
        node.publish_state=Mock();node.start_observing=Mock(side_effect=lambda:setattr(node,'phase','observing'))
        node.face_candidate_visible=Mock(return_value=True)
        node.visual_packet=SimpleNamespace(people=[SimpleNamespace(position=SimpleNamespace(x=1.,y=2.))])
        t=TransformStamped();t.header.stamp.sec=100;t.transform.rotation.w=1.
        node.seconds=lambda:100.;node.tf=SimpleNamespace(lookup_transform=lambda *args:t)
        node.spin_client=Mock();node.event=Mock();node.confirm_caller=Mock()
        with tempfile.TemporaryDirectory() as folder,patch('mission.DATA',Path(folder)),patch.object(Approach,'tick'):
            node.tick()
        self.assertEqual(node.phase,'turn_acceptance');self.assertTrue(node.face_centering)
        self.assertAlmostEqual(node.spin_client.send_goal_async.call_args.args[0].target_yaw,math.atan2(2.,1.))
        node.confirm_caller.assert_not_called()

    def test_nav2_collision_records_actual_code_and_enters_recovery(self):
        node=FaceVoiceSearch.__new__(FaceVoiceSearch);node.data={};node.phase='turning'
        node.cancel_requested=False
        result=Spin.Result(error_code=Spin.Result.COLLISION_AHEAD,error_msg='Collision Ahead')
        node.pending=SimpleNamespace(result=lambda:SimpleNamespace(result=result))
        node.recover_turn_collision=Mock(return_value=True)
        with patch.object(VoiceSearch,'abort') as abort:
            node.abort('Nav2 voice mission action failed:6')
            abort.assert_not_called();node.recover_turn_collision.assert_called_once()
        self.assertEqual(node.data['navigation_failure']['error_code'],703)

    def recovery_node(self):
        node=FaceVoiceSearch.__new__(FaceVoiceSearch)
        node.phase='turning';node.data={};node.collision_retries=0;node.cancel_requested=False;node.done=False
        node.seconds=lambda:100.;node.event=Mock();node.tf=SimpleNamespace(lookup_transform=lambda *args:TransformStamped())
        node.clear_clients=[Mock(),Mock()];node.turn_target_bearing=.8
        for client in node.clear_clients:
            future=Future();future.set_result(None);client.call_async.return_value=future
        node.costmap_stamps={};node.spin_client=Mock();node.start_observing=Mock()
        return node

    def test_refresh_before_retry_uses_remaining_angle(self):
        node=self.recovery_node();node.recover_turn_collision()
        self.assertEqual(node.phase,'collision_refresh');self.assertIsNone(node.nav_handle)
        self.assertEqual(node.collision_retries,1)
        for c in node.clear_clients:c.call_async.assert_called_once()
        node.collision_wait_until=time.monotonic()-1;node.collision_deadline=time.monotonic()+5
        node.tick_collision_recovery();node.spin_client.send_goal_async.assert_not_called()
        node.seconds=lambda:101.;node.costmap_stamps={'local':100.8,'global':100.9}
        t=TransformStamped();t.header.stamp.sec=101;t.transform.rotation.z=__import__('math').sin(.3/2);t.transform.rotation.w=__import__('math').cos(.3/2)
        node.tf.lookup_transform=lambda *args:t
        node.tick_collision_recovery()
        goal=node.spin_client.send_goal_async.call_args.args[0]
        self.assertAlmostEqual(goal.target_yaw,.5,places=5);self.assertEqual(node.phase,'turn_acceptance')

    def test_second_collision_uses_room_planner_without_blind_spin(self):
        node=self.recovery_node();node.collision_retries=1
        node.confirm_caller=Mock(return_value=False);node.choose_room=Mock(return_value=True)
        node.recover_turn_collision()
        node.choose_room.assert_called_once();node.spin_client.send_goal_async.assert_not_called()
        self.assertTrue(node.skip_room_scan);self.assertFalse(node.reached_room_for_scan)
        self.assertEqual(len(node.data['collision_recovery']),1)

    def test_stale_costmaps_or_localization_never_send_motion(self):
        node=self.recovery_node();node.recover_turn_collision();node.abort=Mock()
        node.collision_wait_until=0;node.collision_deadline=0
        node.tick_collision_recovery();node.abort.assert_called_once();node.spin_client.send_goal_async.assert_not_called()
        node.abort.reset_mock();node.seconds=lambda:102.;node.costmap_stamps={'local':101.8,'global':101.9}
        node.tick_collision_recovery();node.abort.assert_called_once();node.spin_client.send_goal_async.assert_not_called()

    def test_interrupt_during_refresh_never_sends_retry(self):
        node=self.recovery_node();node.recover_turn_collision();node.cancel_requested=True
        node.tick_collision_recovery();node.spin_client.send_goal_async.assert_not_called()

    def test_other_failures_are_not_retried_as_collision(self):
        node=self.recovery_node();node.pending=SimpleNamespace(result=lambda:SimpleNamespace(result=Spin.Result(error_code=702)))
        with patch.object(VoiceSearch,'abort') as abort:
            node.abort('Nav2 voice mission action failed:6');abort.assert_called_once()
        self.assertFalse(node.clear_clients[0].call_async.called)

    def test_new_room_selection_waits_for_real_arrival_before_scan(self):
        node=FaceVoiceSearch.__new__(FaceVoiceSearch);node.reached_room_for_scan=True
        node.identity=None
        node.face_reacquire_until=0.;node.scan_count=0
        node.room_scans={};node.room_attempts={};node.visited=set();node.data={}
        node.rooms=SimpleNamespace(rooms=[SimpleNamespace(id='west',outline=SimpleNamespace(points=[SimpleNamespace(x=x,y=y) for x,y in [(-6,-2),(-2,-2),(-2,2),(-6,2)]]))])
        node.room_client=Mock();node.room_client.service_is_ready.return_value=True
        node.robot=lambda:np.array([0.,0.]);node.event=Mock()
        node.refresh_prediction=lambda:{'visual':None,'room_candidates':[{'id':'west','area_m2':16,'audio_angle_error_deg':180.,'priority':.2}]}
        self.assertTrue(node.choose_room())
        self.assertEqual(node.room_id,'west')
        self.assertFalse(node.reached_room_for_scan)

    def test_same_caller_refreshes_evidence_without_task_preemption(self):
        import json
        from std_msgs.msg import String
        node=FaceVoiceSearch.__new__(FaceVoiceSearch)
        node.done=False;node.cancel_requested=False;node.identity='caller';node.face_ids=['caller','other']
        node.calls=[{'stamp':99.}];node.seconds=lambda:100.;node.publish_state=Mock()
        node.phase='search_walking';node.pending=object();node.pending_audio=None
        pending=node.pending
        call=lambda who:node.call(String(data=json.dumps({'identity':who,'keyword':'小北小北','stamp':99.5,'azimuth_rad':1.})))
        with patch('mission.read_json',return_value={'ready':True}):
            call('other');self.assertIsNone(node.pending_audio)
            call('caller');self.assertIsNotNone(node.pending_audio)
        self.assertEqual(node.identity,'caller');self.assertEqual(node.phase,'search_walking')
        self.assertIs(node.pending,pending)

    def test_face_reacquisition_cannot_launch_another_room_route(self):
        node=FaceVoiceSearch.__new__(FaceVoiceSearch);node.face_reacquire_until=time.monotonic()+20
        with patch.object(VoiceSearch,'choose_room') as choose:
            self.assertFalse(node.choose_room());choose.assert_not_called()

    def test_planner_no_path_moves_to_next_observed_room_not_arrival(self):
        node=self.recovery_node();node.phase='search_walking';node.room_id='R2';node.visited=set()
        node.start_clearance_recovery=Mock(return_value=False)
        node.pending=SimpleNamespace(result=lambda:SimpleNamespace(result=ComputePathToPose.Result(error_code=208)))
        node.face_history=[]
        # Exercise the real wrapper: it must not mistake the failed route for
        # successful arrival when start_observing inspects the previous phase.
        node.start_observing=FaceVoiceSearch.start_observing.__get__(node)
        with patch.object(VoiceSearch,'abort') as abort,patch.object(VoiceSearch,'start_observing') as observe:
            node.abort('Nav2 voice mission action failed:6');abort.assert_not_called()
            observe.assert_called_once()
        self.assertEqual(node.visited,{'R2'});self.assertFalse(node.reached_room_for_scan)
        self.assertEqual(node.data['unreachable_rooms'][0]['error_code'],208)
        self.assertNotIn('visited_rooms',node.data)

    def test_clear_service_error_never_retries_motion(self):
        node=self.recovery_node();node.recover_turn_collision();node.abort=Mock()
        bad=Future();bad.set_exception(RuntimeError('service disconnected'));node.clear_pending=[bad]
        node.collision_wait_until=0;node.collision_deadline=time.monotonic()+5
        node.seconds=lambda:101.;node.costmap_stamps={'local':100.9,'global':100.9}
        node.tick_collision_recovery();node.abort.assert_called_once();node.spin_client.send_goal_async.assert_not_called()

    def face_node(self):
        node=self.recovery_node();node.phase='walking';node.face_retries=0
        node.target=np.array([2.,1.]);node.nav_handle=Mock(accepted=True)
        node.nav_result=Future();node.approaching=True
        node.last_seen=time.monotonic()-2.1
        return node

    def test_lost_face_must_acknowledge_cancel_before_reorientation(self):
        node=self.face_node()
        with patch.object(VoiceSearch,'abort') as abort:
            node.abort('visual track lost or stale; cancel navigation');abort.assert_not_called()
        self.assertEqual(node.phase,'face_recovery_stop');node.nav_handle.cancel_goal_async.assert_called_once()
        node.tick_face_recovery();node.spin_client.send_goal_async.assert_not_called()
        self.assertTrue(node.approaching)
        node.nav_result.set_result(SimpleNamespace(status=GoalStatus.STATUS_CANCELED))
        t=TransformStamped();t.header.stamp.sec=100;t.transform.rotation.w=1.
        node.tf.lookup_transform=lambda *args:t
        node.tick_face_recovery()
        self.assertFalse(node.approaching);self.assertIsNone(node.locked)
        self.assertEqual(node.phase,'turn_acceptance')
        goal=node.spin_client.send_goal_async.call_args.args[0]
        self.assertAlmostEqual(goal.target_yaw,__import__('math').atan2(1.,2.))
        self.assertGreater(node.face_reacquire_until,time.monotonic())

    def test_cancel_or_exhausted_face_recovery_never_retries(self):
        node=self.face_node();node.face_retries=2
        with patch.object(VoiceSearch,'abort') as abort:
            node.abort('visual track lost or stale; cancel navigation');abort.assert_called_once()
        node.nav_handle.cancel_goal_async.assert_not_called()
        node.cancel_requested=True
        node.tick_face_recovery();node.spin_client.send_goal_async.assert_not_called()

    def test_viewpoint_search_uses_recent_measurement_and_bounded_standoff(self):
        from nav_msgs.msg import OccupancyGrid
        node=FaceVoiceSearch.__new__(FaceVoiceSearch)
        node.face_view_attempts=0;node.face_observed_wall=time.monotonic()-.3
        node.nav=Mock();node.nav.server_is_ready.return_value=True
        node.map=OccupancyGrid();node.map.info.width=node.map.info.height=120
        node.map.info.resolution=.1;node.map.info.origin.position.x=node.map.info.origin.position.y=-6.
        node.map.info.origin.orientation.w=1.;node.map.data=[0]*14400
        node.robot=lambda:np.zeros(2);node.face_last_position=np.array([2.,0.])
        node.face_ids=[];node.identity='caller';node.data={};node.seconds=lambda:100.;node.event=Mock()
        node.pose=lambda goal:__import__('geometry_msgs.msg',fromlist=['PoseStamped']).PoseStamped()
        self.assertTrue(node.start_face_viewpoint())
        row=node.data['face_view_searches'][0]
        self.assertGreaterEqual(np.linalg.norm(np.asarray(row['goal'][:2])-node.face_last_position),1.59)
        self.assertLessEqual(row['hint_distance_m'],4.)
        node.nav.reset_mock();node.face_observed_wall=time.monotonic()-16
        self.assertFalse(node.start_face_viewpoint());node.nav.send_goal_async.assert_not_called()

    def test_interrupt_pending_viewpoint_must_cancel_accepted_handle(self):
        node=FaceVoiceSearch.__new__(FaceVoiceSearch)
        node.phase='face_view_acceptance';node.cancel_requested=False;node.done=False
        node.data={};node.event=Mock();node.nav_handle=None
        node.pending=Future();handle=Mock(accepted=True)
        node.abort('user interrupted; safe stop')
        self.assertTrue(node.cancel_requested);self.assertFalse(node.done)
        with patch.object(Approach,'tick'):
            node.tick_face_viewpoint();handle.cancel_goal_async.assert_not_called()
            node.pending.set_result(handle);node.tick_face_viewpoint()
        handle.cancel_goal_async.assert_called_once();self.assertEqual(node.phase,'cancelling')

    def test_nearby_remembered_face_is_inspected_before_excluding_current_room(self):
        node=FaceVoiceSearch.__new__(FaceVoiceSearch)
        node.identity='caller';node.data={};node.robot=lambda:np.zeros(2)
        node.memory=Mock();node.memory.visual_prior.return_value={'age_s':35.,'position':[2.,1.],
            'source':'camera face ROI + aligned depth + measurement-time map TF'}
        node.start_face_viewpoint=Mock(return_value=True);node.event=Mock()
        self.assertTrue(node.try_nearby_memory_view())
        node.start_face_viewpoint.assert_called_once_with(max_age_s=60.)
        self.assertTrue(node.data['remembered_view_search']['final_acceptance_requires_fresh_face'])
        self.assertFalse(node.try_nearby_memory_view())
        node.data={};node.memory.visual_prior.return_value['age_s']=61.
        node.start_face_viewpoint.reset_mock()
        self.assertFalse(node.try_nearby_memory_view());node.start_face_viewpoint.assert_not_called()


    def test_already_beside_uses_existing_stop_check_without_unnecessary_path(self):
        node=FaceVoiceSearch.__new__(FaceVoiceSearch)
        node.phase='planning';node.locked=3;node.target=np.array([1.2,0.]);node.robot=lambda:np.zeros(2)
        node.nav_result=None;node.data={};node.event=Mock();node.pending=None
        self.assertTrue(node.accept_already_beside());self.assertEqual(node.phase,'checking_stop')
        self.assertIsNone(node.settle_since)
        node.phase='planning';node.target=np.array([3.,0.]);self.assertFalse(node.accept_already_beside())
        node.target=np.array([1.2,0.]);node.nav_result=Future()
        self.assertFalse(node.accept_already_beside());self.assertEqual(node.phase,'planning')


if __name__=='__main__':unittest.main()
