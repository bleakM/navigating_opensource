"""Body appearance remains a fallible hint and cannot replace a fresh face."""
import math,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from body_identity import match_body,torso_pixel,pose_attributes
from caller_memory import CallerMemory,bearing_hypotheses


class BodyTests(unittest.TestCase):
    def test_appearance_requires_absolute_score_and_runner_up_margin(self):
        gallery={'a':{'embeddings':[[1,0,0]]},'b':{'embeddings':[[0,1,0]]}}
        self.assertEqual(match_body([1,0,0],gallery)[0],'a')
        self.assertIsNone(match_body([1,1,0],gallery)[0])
        self.assertIsNone(match_body([0,0,1],gallery)[0])
        self.assertIsNone(match_body([1,0,0],{})[0])

    def test_body_does_not_overwrite_recent_face_and_expires_sooner(self):
        with tempfile.TemporaryDirectory() as folder:
            memory=CallerMemory(Path(folder)/'m.json','map')
            memory.observe_face('a',[1,1],1,now=100)
            memory.observe_body('a',[5,5],2,now=102)
            self.assertEqual(memory.visual_prior('a',now=103)['identity_evidence'],'face')
            p=memory.visual_prior('a',now=107)
            self.assertEqual(p['identity_evidence'],'body_hint');self.assertEqual(p['position'],[5,5])
            self.assertLess(p['weight'],.55);self.assertIsNone(memory.visual_prior('a',now=150))
            self.assertFalse(memory.observe_body('a',[float('nan'),0],3,now=151))

    def test_secondary_audio_peak_uses_capture_rotation_once(self):
        call={'map_azimuth':math.pi/2,'array_azimuth':0.,'angular_candidates':[
            {'azimuth_rad':0.,'relative_score':1.},
            {'azimuth_rad':math.pi/2,'relative_score':.9},
            {'azimuth_rad':-math.pi/2,'relative_score':.1}]}
        bands=bearing_hypotheses(call)
        self.assertEqual(len(bands),2)
        self.assertAlmostEqual(bands[1]['map_azimuth'],math.pi)

    def test_ambiguous_rooms_keep_reserves_instead_of_hiding_uncertainty(self):
        with tempfile.TemporaryDirectory() as folder:
            memory=CallerMemory(Path(folder)/'m.json','map')
            rooms=[{'id':str(i),'centroid':[i+3,0],'area':10,'outline':[],'doorways':[]} for i in range(5)]
            result=memory.predict('a',{'origin':[0,0],'map_azimuth':0.,'stamp':1},rooms,now=100)
            self.assertEqual(len(result['primary_room_ids']),3)
            self.assertEqual(len(result['reserve_room_ids']),2)
            self.assertTrue(result['acoustically_ambiguous']);self.assertFalse(result['primary_is_guaranteed'])

    def test_receiver_corridor_kept_local_without_using_remote_room_slot(self):
        with tempfile.TemporaryDirectory() as folder:
            memory=CallerMemory(Path(folder)/'m.json','map')
            rooms=[dict(id='corridor',centroid=[0,0],area=20,outline=[[-8,-1],[8,-1],[8,1],[-8,1]],doorways=[])]
            rooms += [dict(id=str(i),centroid=[4+i,3],area=10,outline=[],doorways=[]) for i in range(4)]
            result=memory.predict('a',{'origin':[0,0],'map_azimuth':0.,'stamp':1},rooms,now=1)
            self.assertEqual(result['local_region_ids'],['corridor'])
            self.assertNotIn('corridor',result['primary_room_ids'])
            self.assertEqual(len(result['primary_room_ids']),3)
            self.assertEqual(len(result['reserve_room_ids']),1)

    def test_depth_target_prefers_observed_torso_keypoints(self):
        kp=np.zeros((17,3));kp[5]=[20,30,.9];kp[6]=[30,30,.9]
        np.testing.assert_allclose(torso_pixel({'keypoints':kp.tolist(),'box':[0,0,60,120]}),[25,30])

    def test_metric_spans_require_visible_landmarks_and_valid_depth(self):
        kp=np.zeros((17,3));kp[0]=[50,10,.9]
        for i,u,v in [(5,40,25),(6,60,25),(11,44,50),(12,56,50),(15,44,85),(16,56,85)]:kp[i]=[u,v,.9]
        depth=np.ones((100,100),np.float32)
        result=pose_attributes(kp,depth,(100,100,50,50))
        self.assertAlmostEqual(result['shoulder_width_m'],.2)
        self.assertAlmostEqual(result['nose_ankle_span_m'],.75)
        self.assertAlmostEqual(result['trunk_leg_ratio'],.25/.35)
        self.assertEqual(pose_attributes(kp,depth*np.nan,(100,100,50,50)),{})


if __name__=='__main__':unittest.main()
