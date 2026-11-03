"""Small RGB-D and approach adapters around pretrained perception and Nav2."""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt


def raised_hand(keypoints, confidence=.45):
    """COCO17 wrists must be visibly above shoulders; missing points never confirm."""
    k=np.asarray(keypoints)
    if k.shape!=(17,3):
        return False
    for shoulder,elbow,wrist,hip in [(5,7,9,11),(6,8,10,12)]:
        if np.min(k[[shoulder,elbow,wrist],2])<confidence:
            continue
        torso=abs(k[hip,1]-k[shoulder,1]) if k[hip,2]>=confidence else 80
        if k[wrist,1]<k[shoulder,1]-max(12,.25*torso):
            return True
    return False


def torso_point(depth, keypoints, intrinsic):
    """Measure central torso depth, rejecting holes/background mixtures."""
    k=np.asarray(keypoints)
    valid=k[[5,6,11,12]]
    valid=valid[valid[:,2]>=.4]
    if len(valid)<3:
        raise ValueError('Torso landmarks unavailable')
    u,v=np.mean(valid[:,:2],axis=0)
    h,w=depth.shape
    if not (8<=u<w-8 and 8<=v<h-8):
        raise ValueError('Torso outside image')
    patch=depth[int(v)-6:int(v)+7,int(u)-6:int(u)+7]
    values=patch[np.isfinite(patch)&(patch>.4)&(patch<7.)]
    if len(values)<40:
        raise ValueError('Insufficient valid depth')
    z=float(np.median(values))
    if np.percentile(np.abs(values-z),80)>.25:
        raise ValueError('Depth discontinuity at torso')
    fx,fy,cx,cy=intrinsic
    return np.array([(u-cx)*z/fx,(v-cy)*z/fy,z])


def safe_approaches(person, robot, grid, resolution, origin, clearance=.55):
    """Candidate endpoints only. Nav2 owns path planning and motion."""
    free=np.asarray(grid)==0
    distance=distance_transform_edt(np.pad(free,1))*resolution
    distance=distance[1:-1,1:-1]
    initial=math.atan2(robot[1]-person[1],robot[0]-person[0])
    yaw=origin[2] if len(origin)>2 else 0.
    c,s=math.cos(yaw),math.sin(yaw)
    for radius in (1.35,1.6,1.85):
        for offset in (0,.35,-.35,.7,-.7,1.2,-1.2,math.pi):
            angle=initial+offset
            x,y=person[0]+radius*math.cos(angle),person[1]+radius*math.sin(angle)
            dx,dy=x-origin[0],y-origin[1]
            col=int(math.floor((c*dx+s*dy)/resolution))
            row=int(math.floor((-s*dx+c*dy)/resolution))
            if 0<=row<free.shape[0] and 0<=col<free.shape[1] and distance[row,col]>=clearance:
                yield (x,y,math.atan2(person[1]-y,person[0]-x))
