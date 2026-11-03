"""Terminal Nav2 failures cannot cause overlapping motion or blind arrival."""
import sys,time,unittest
from pathlib import Path
from concurrent.futures import Future
from types import SimpleNamespace as S
from unittest.mock import Mock,patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mission import FaceVoiceSearch
from approach_person import Approach
from action_msgs.msg import GoalStatus
from nav2_msgs.action import FollowPath
from geometry_msgs.msg import TransformStamped

class TerminalTests(unittest.TestCase):
    def node(self,distance=1.8):
        n=FaceVoiceSearch.__new__(FaceVoiceSearch)
        n.phase='walking';n.cancel_requested=False;n.done=False;n.data={}
        n.nav_result=Future();n.nav_result.set_result(S(status=GoalStatus.STATUS_ABORTED,
            result=S(error_code=FollowPath.Result.PATIENCE_EXCEEDED,error_msg='patience')))
        n.nav_handle=Mock(accepted=True);n.pending=n.nav_result
        n.people=S(header=S(stamp=S(sec=100,nanosec=0)),people=[S(track_id=8,confidence=.7)])
        n.locked=8;n.last_seen=time.monotonic();n.seconds=lambda:100.1
        n.goal=[1.,2.,0.];n.target=np.array([0.,distance]);n.robot=lambda:np.zeros(2)
        n.approach_retries=0;n.event=Mock();n.candidates=iter([[3.,4.,0.]])
        t=TransformStamped();t.header.stamp.sec=100
        n.tf=S(lookup_transform=lambda *args:t)
        return n

    def test_terminal_at_person_only_begins_strict_stop_verification(self):
        n=self.node();self.assertTrue(n.recover_approach_terminal())
        self.assertEqual(n.phase,'checking_stop');self.assertIsNone(n.nav_handle)
        self.assertIsNone(n.pending);self.assertFalse(n.done);self.assertNotIn('passed',n.data)

    def test_pending_cancelled_and_user_cancel_never_accept_or_retry(self):
        for state in ('pending','cancelled','user_cancel'):
            n=self.node()
            if state=='pending':n.nav_result=Future()
            elif state=='cancelled':n.nav_result.result().status=GoalStatus.STATUS_CANCELED
            else:n.cancel_requested=True
            self.assertFalse(n.recover_approach_terminal());self.assertEqual(n.phase,'walking')

    def test_stale_missing_or_wrong_face_cannot_accept_arrival(self):
        for state in ('old','empty','wrong','low','old_stamp'):
            n=self.node()
            if state=='old':n.last_seen-=2
            elif state=='empty':n.people.people=[]
            elif state=='wrong':n.people.people[0].track_id=2
            elif state=='low':n.people.people[0].confidence=.2
            else:n.people.header.stamp.sec=97
            self.assertFalse(n.recover_approach_terminal());self.assertNotEqual(n.phase,'checking_stop')

    def test_next_endpoint_is_bounded_and_does_not_reset_iterator(self):
        n=self.node(3.);self.assertTrue(n.recover_approach_terminal())
        self.assertEqual(n.phase,'planning');self.assertEqual(n.approach_retries,1)
        self.assertEqual(next(n.candidates),[3.,4.,0.]);self.assertIsNone(n.nav_handle)
        n=self.node(3.);n.approach_retries=2
        self.assertFalse(n.recover_approach_terminal())

    def test_stale_tf_and_unknown_error_fail_safely(self):
        n=self.node();n.seconds=lambda:102.
        self.assertFalse(n.recover_approach_terminal())
        n=self.node();n.nav_result.result().result.error_code=9999
        self.assertFalse(n.recover_approach_terminal())

    def test_stop_check_still_requires_three_seconds_and_valid_distance(self):
        n=self.node();self.assertTrue(n.recover_approach_terminal())
        n.lifecycle_pending={};n.lifecycle_checked=time.monotonic();n.deadline=time.monotonic()+20
        n.odometry=[(0.,0.,0.,0.)];n.finish=Mock();n.abort=Mock()
        Approach.tick(n);n.finish.assert_not_called()
        n.settle_since=time.monotonic()-3.1
        Approach.tick(n);n.finish.assert_called_once_with(True)
        n.finish.reset_mock();n.target=np.array([0.,2.1])
        Approach.tick(n);n.finish.assert_not_called();n.abort.assert_called_once()

if __name__=='__main__':unittest.main()
