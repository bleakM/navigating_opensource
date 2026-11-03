"""Check physical command bounds, coordination and stopping without a leg policy."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arm_motion import ArmMotion,JOINTS

class ArmSafety(unittest.TestCase):
    def test_bounded_speed_acceleration_under_phase_reversal(self):
        states={j:0. for j in JOINTS};motion=ArmMotion(states);old=[0.]*len(JOINTS)
        for n in range(1000):
            states.update(left_hip_pitch_joint=(-1.)**(n//50),right_hip_pitch_joint=-(-1.)**(n//50))
            p,v=motion.update(states,{},.6,.9,.02)
            self.assertLessEqual(max(abs(x) for x in v),1.5+1e-9)
            self.assertLessEqual(max(abs(x-y)/.02 for x,y in zip(v,old)),6.+1e-9);old=v
        self.assertTrue(all(abs(x)<.5 for x in p))

    def test_opposite_arm_and_return_to_rest(self):
        states={j:0. for j in JOINTS};states.update(left_hip_pitch_joint=.2,right_hip_pitch_joint=-.2)
        motion=ArmMotion(states)
        for _ in range(250):p,v=motion.update(states,{},.6,0.,.02)
        self.assertLess(p[0],p[7]);self.assertGreater(p[3],.25)
        for _ in range(250):p,v=motion.update(states,{},0.,0.,.02)
        self.assertAlmostEqual(p[0],.02,places=5);self.assertAlmostEqual(p[7],.02,places=5)
        self.assertLess(max(abs(x) for x in v),1e-5)

if __name__=='__main__':unittest.main()
