import sys,unittest
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[2]/'interaction/src')]
from face_approach import ranked_face_approaches
from vision_geometry import safe_approaches


class FaceViewTests(unittest.TestCase):
    def setUp(self):
        self.person=np.array([2.,0.]);self.robot=np.array([-2.,0.])
        self.grid=np.zeros((160,160),dtype=np.int8);self.origin=[-8.,-8.,0.]
    def goals(self,others):
        return ranked_face_approaches(self.person,self.robot,self.grid,.1,self.origin,others)
    def test_occluding_measured_face_prefers_lateral_view(self):
        goals=self.goals([[1.,0.]])
        self.assertGreater(abs(goals[0][1]),.35)
        # Ranking must not introduce any unchecked endpoint geometry.
        original=set(safe_approaches(self.person,self.robot,self.grid,.1,self.origin))
        self.assertEqual(set(goals),original)
    def test_no_other_faces_keeps_nearest_approach(self):
        self.assertAlmostEqual(self.goals([])[0][1],0.,places=5)
    def test_no_safe_endpoint_remains_empty(self):
        self.grid[:]=100;self.assertEqual(self.goals([[1.,0.]]),[])
    def test_final_endpoints_allow_nav2_tolerance_inside_person_acceptance(self):
        goals=ranked_face_approaches(self.person,self.robot,self.grid,.1,self.origin,[[1.,0.]],max_standoff=1.65)
        self.assertTrue(goals)
        for goal in goals:
            distance=np.linalg.norm(np.asarray(goal[:2])-self.person)
            self.assertGreaterEqual(distance-.25,1.)
            self.assertLessEqual(distance+.25,1.95)

if __name__=='__main__':unittest.main()
