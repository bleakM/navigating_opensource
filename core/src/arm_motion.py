"""Bounded upper-body coordination driven by measured hip phase, not a gait policy."""
import math

PARTS=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']
JOINTS=[f'{side}_{part}_joint' for side in ('left','right') for part in PARTS]


class ArmMotion:
    max_speed=1.5
    max_acceleration=6.0

    def __init__(self,positions):
        self.target=[positions[name] for name in JOINTS]
        self.velocity=[0.0]*len(JOINTS)
        self.swing=0.0

    def update(self,positions,velocities,linear,angular,dt):
        dt=max(.001,min(.04,dt))
        phase=positions.get('left_hip_pitch_joint',0.)-positions.get('right_hip_pitch_joint',0.)
        phase_rate=velocities.get('left_hip_pitch_joint',0.)-velocities.get('right_hip_pitch_joint',0.)
        # Small phase lead compensates trajectory latency. The opposite arm
        # follows the measured leg; there is no free-running arm oscillator.
        walking=min(1.,max(0.,(abs(linear)-.06)/.20))
        turn_scale=1./(1.+.55*abs(angular))
        self.swing=max(-.32,min(.32,-1.10*(phase+.06*phase_rate)))*walking*turn_scale
        s=self.swing
        desired=[.02+s,.07,0.,.28+.12*max(0.,-s),0.,.02,0.,
                 .02-s,-.07,0.,.28+.12*max(0.,s),0.,-.02,0.]
        for i,goal in enumerate(desired):
            acceleration=max(-self.max_acceleration,min(self.max_acceleration,
                100.*(goal-self.target[i])-20.*self.velocity[i]))
            self.velocity[i]=max(-self.max_speed,min(self.max_speed,self.velocity[i]+acceleration*dt))
            self.target[i]+=self.velocity[i]*dt
        return self.target[:],self.velocity[:]
