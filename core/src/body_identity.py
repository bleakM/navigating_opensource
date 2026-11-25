"""Pretrained OSNet appearance hints. These never constitute face confirmation.

Online inputs are camera pixels, pose keypoints and registered appearance
templates. No simulator actor state or scene coordinates are used here.
"""
import sys,time
from pathlib import Path
import cv2,numpy as np
from common import ROOT
from identity import unit


def colour_descriptor(crop):
    """OpenCV HSV clothing bands, a low-weight companion to OSNet.

    Central bands reduce background contamination. This is appearance within
    this session, not a clothing-invariant biometric identity measurement.
    """
    image=cv2.cvtColor(cv2.resize(crop,(64,128)),cv2.COLOR_BGR2HSV)
    result=[]
    for y0,y1 in ((12,48),(48,86),(86,124)):
        histogram=cv2.calcHist([image[y0:y1,12:52]],[0,1],None,[12,4],[0,180,0,256]).ravel()
        result.extend(np.sqrt(histogram/max(float(histogram.sum()),1.)).tolist())
    return unit(result)


def match_body(vector,gallery,threshold=.68,margin=.06,colour=None):
    scores={}
    for name,row in gallery.items():
        if not row.get('embeddings'):continue
        feature=max(float(unit(vector)@unit(v)) for v in row['embeddings'])
        colours=row.get('colours',[])
        support=max(float(unit(colour)@unit(v)) for v in colours) if colour is not None and colours else None
        # Never let colour rescue a genuinely incompatible neural descriptor.
        scores[name]=(.75*feature+.25*support if support is not None and feature>=.55 else feature)
    ranked=sorted(scores,key=scores.get,reverse=True)
    if not ranked:return None,0.,0.,scores
    best=ranked[0];gap=scores[best]-(scores[ranked[1]] if len(ranked)>1 else -1.)
    return (best if scores[best]>=threshold and gap>=margin else None),scores[best],gap,scores


def pose_attributes(keypoints,depth,intrinsics):
    """Measured body spans, never an assumed asset height or hidden pose.

    Missing/occluded landmarks yield no measurement. Nose-to-ankle span is
    deliberately not labelled total height, which this camera does not measure.
    """
    kp=np.asarray(keypoints,float);out={}
    if kp.shape!=(17,3):return out
    points={}
    for i in (0,5,6,11,12,15,16):
        u,v,confidence=kp[i]
        if confidence<.65:continue
        u,v=int(round(u)),int(round(v))
        if not 3<=u<depth.shape[1]-3 or not 3<=v<depth.shape[0]-3:continue
        patch=depth[v-2:v+3,u-2:u+3];valid=patch[np.isfinite(patch)&(patch>.4)&(patch<9)]
        if len(valid)<12:continue
        z=float(np.median(valid));fx,fy,cx,cy=intrinsics
        points[i]=np.array([(u-cx)*z/fx,(v-cy)*z/fy,z])
    if all(i in points for i in (5,6)):
        span=float(np.linalg.norm(points[5]-points[6]))
        if .15<span<.75:out['shoulder_width_m']=span
    ankles=[points[i] for i in (15,16) if i in points]
    if 0 in points and ankles:
        span=float(np.linalg.norm(points[0]-np.mean(ankles,axis=0)))
        if .55<span<2.1:out['nose_ankle_span_m']=span
    if all(i in points for i in (5,6,11,12)) and ankles:
        shoulders=(points[5]+points[6])/2;hips=(points[11]+points[12])/2
        legs=float(np.linalg.norm(hips-np.mean(ankles,axis=0)))
        if legs>.3:out['trunk_leg_ratio']=float(np.linalg.norm(shoulders-hips)/legs)
    return out


class BodyEncoder:
    def __init__(self,weights=None):
        import torch
        sys.path.insert(0,str(ROOT/'core/vendor'))
        from osnet import osnet_x0_25
        self.torch=torch;torch.set_num_threads(2)
        self.model=osnet_x0_25(num_classes=1041,pretrained=False).eval()
        state=torch.load(weights or ROOT/'models/osnet_x0_25_msmt17.pt',map_location='cpu',weights_only=True)
        # Classifier is unused at inference. Require every feature tensor to
        # match; a wrong checkpoint must fail rather than use random features.
        state={k.removeprefix('module.'):v for k,v in state.items()}
        state={k:v for k,v in state.items() if not k.startswith('classifier.')}
        missing,unexpected=self.model.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith('classifier.') for k in missing):
            raise ValueError(f'OSNet checkpoint mismatch: {missing}, {unexpected}')

    def encode(self,crops):
        if not crops:return []
        batch=np.stack([cv2.resize(cv2.cvtColor(c,cv2.COLOR_BGR2RGB),(128,256)) for c in crops]).astype('float32')/255.
        batch=(batch-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        x=self.torch.from_numpy(batch.transpose(0,3,1,2).copy())
        with self.torch.inference_mode():features=self.model(x).numpy()
        return [unit(v) for v in features]


class Bodies:
    def __init__(self):
        from ultralytics import YOLO
        self.encoder=BodyEncoder()
        self.detector=YOLO(str(ROOT/'models/yolo11n-pose.pt'))

    def identify(self,image,gallery):
        started=time.monotonic()
        result=self.detector.predict(image,conf=.45,imgsz=640,classes=[0],verbose=False,device='cpu')[0]
        # Ultralytics initializes its CPU backend with its own thread count on
        # the first prediction. Restore our real-time budget before OSNet and
        # subsequent frames; extra workers compete with physics and SLAM.
        self.encoder.torch.set_num_threads(2)
        rows=[];crops=[];height,width=image.shape[:2]
        for i,box in enumerate(result.boxes.xyxy.cpu().numpy()):
            x0,y0,x1,y1=np.rint(box).astype(int);x0=max(0,x0);y0=max(0,y0);x1=min(width,x1);y1=min(height,y1)
            if y1-y0<80 or x1-x0<24:continue
            kp=result.keypoints.data[i].cpu().numpy().tolist() if result.keypoints is not None else []
            rows.append({'box':list(map(int,[x0,y0,x1,y1])),'keypoints':kp,'detector_score':float(result.boxes.conf[i]),
                         'full_body_visible':bool(y0>2 and y1<height-3)})
            crops.append(image[y0:y1,x0:x1])
        features=self.encoder.encode(crops)
        for row,feature,crop in zip(rows,features,crops):
            colour=colour_descriptor(crop)
            who,score,margin,scores=match_body(feature,gallery,colour=colour)
            row.update(identity=who,score=score,margin=margin,scores=scores,
                       embedding=feature.tolist(),colour=colour.tolist())
        return rows,time.monotonic()-started


def torso_pixel(row):
    """Visible shoulder/hip keypoints give a depth patch away from background."""
    kp=np.asarray(row.get('keypoints',[]),float)
    if kp.shape==(17,3):
        torso=kp[[5,6,11,12]];torso=torso[torso[:,2]>.5]
        if len(torso)>=2:return np.median(torso[:,:2],axis=0)
    x0,y0,x1,y1=row['box'];return np.array([(x0+x1)*.5,y0+(y1-y0)*.4])


def initialize_body_worker(gallery):
    # A separate interpreter prevents high-rate ROS/TF callbacks from holding
    # the GIL between the many small pretrained network operations.
    global _body_worker_model,_body_worker_gallery
    cv2.setNumThreads(1)
    _body_worker_model=Bodies();_body_worker_gallery=gallery


def body_worker_ready():return True


def infer_body_frame(image):
    rows,latency=_body_worker_model.identify(image,_body_worker_gallery)
    for row in rows:
        row.pop('embedding',None);row.pop('colour',None)
    return rows,latency
