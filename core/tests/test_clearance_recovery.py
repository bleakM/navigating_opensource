"""Recovery retains exclusive action ownership and cancellation acknowledgement."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import Mock,patch
from action_msgs.msg import GoalStatus
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mission import FaceVoiceSearch

class ClearanceTests(unittest.TestCase):
    def state(self):
        return S(clearance_attempts={},room_id='room-a',seconds=lambda:10.,data={},
            clearance_client=Mock(),nav_handle=None,nav_result=None,pending=None,
            phase='search_walking',event=Mock(),done=False,cancel_requested=False,
            room_client=Mock(),abort=Mock())

    def test_one_retry_per_room_and_total_two(self):
        n=self.state();n.clearance_client.server_is_ready.return_value=True
        self.assertTrue(FaceVoiceSearch.start_clearance_recovery(n,105,'patience'))
        self.assertEqual(n.phase,'clearance_acceptance')
        self.assertIsNone(n.nav_handle)
        self.assertFalse(FaceVoiceSearch.start_clearance_recovery(n,105,'patience'))
        n.room_id='room-b';self.assertTrue(FaceVoiceSearch.start_clearance_recovery(n,104,'path'))
        n.room_id='room-c';self.assertFalse(FaceVoiceSearch.start_clearance_recovery(n,104,'path'))

    @patch('mission.Approach.tick')
    def test_cancel_during_acceptance_cancels_accepted_action(self,_):
        n=self.state();n.phase='clearance_acceptance';n.cancel_requested=True
        h=Mock(accepted=True);n.pending=Mock();n.pending.done.return_value=True;n.pending.result.return_value=h
        FaceVoiceSearch.tick_clearance_recovery(n)
        h.cancel_goal_async.assert_called_once();self.assertEqual(n.phase,'cancelling')
        n.room_client.call_async.assert_not_called()

    @patch('mission.Approach.tick')
    def test_recovery_failure_does_not_mark_room_inspected(self,_):
        n=self.state();n.phase='clearance_walking';n.data['clearance_recovery']=[{}]
        n.nav_result=Mock();n.nav_result.done.return_value=True
        n.nav_result.result.return_value=S(status=GoalStatus.STATUS_ABORTED,
            result=S(success=False,clearance_m=.4,message='blocked'))
        FaceVoiceSearch.tick_clearance_recovery(n)
        n.abort.assert_called_once();n.room_client.call_async.assert_not_called()
        self.assertEqual(n.phase,'clearance_failed')

    @patch('mission.Approach.tick')
    def test_success_retries_original_observed_room(self,_):
        n=self.state();n.phase='clearance_walking';n.data['clearance_recovery']=[{}]
        n.nav_result=Mock();n.nav_result.done.return_value=True
        n.nav_result.result.return_value=S(status=GoalStatus.STATUS_SUCCEEDED,
            result=S(success=True,clearance_m=.7,message='clear'))
        FaceVoiceSearch.tick_clearance_recovery(n)
        request=n.room_client.call_async.call_args.args[0]
        self.assertEqual(request.target,'room-a');self.assertEqual(n.phase,'resolving_room')
        self.assertIsNone(n.nav_handle)

if __name__=='__main__':unittest.main()
