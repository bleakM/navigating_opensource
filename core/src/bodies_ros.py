#!/usr/bin/env python3
"""Latest-frame RGB-D body hints, separate from the face acceptance topic."""
import threading,time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np,rclpy,message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image,CameraInfo
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import Buffer,TransformListener,TransformException
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point
from g1_interaction_msgs.msg import PersonTracks,PersonTrack
from body_identity import torso_pixel,pose_attributes,initialize_body_worker,body_worker_ready,infer_body_frame
from common import DATA,read_json,write_json,atomic


class BodyNode(Node):
    def __init__(self):
        super().__init__('xiaobei_body_recognition',parameter_overrides=[Parameter('use_sim_time',value=True)])
        cv2.setNumThreads(1)
        self.bridge=CvBridge();self.tf=Buffer();self.listener=TransformListener(self.tf,self)
        self.gallery=read_json(DATA/'body-gallery.json',{});self.ids=list(read_json(DATA/'face-gallery.json',{}))
        self.pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_body_worker,initargs=(self.gallery,))
        self.warmup=self.pool.submit(body_worker_ready)
        self.packet=None;self.info=None;self.lock=threading.Lock();self.stop=threading.Event()
        self.pub=self.create_publisher(PersonTracks,'/wb/bodies',10)
        self.create_subscription(CameraInfo,'/interaction_camera/color/camera_info',lambda m:setattr(self,'info',m),qos_profile_sensor_data)
        rgb=message_filters.Subscriber(self,Image,'/interaction_camera/color/image_raw',qos_profile=qos_profile_sensor_data)
        depth=message_filters.Subscriber(self,Image,'/interaction_camera/aligned_depth_to_color/image_raw',qos_profile=qos_profile_sensor_data)
        self.sync=message_filters.ApproximateTimeSynchronizer([rgb,depth],3,.04);self.sync.registerCallback(self.receive)
        self.worker=threading.Thread(target=self.process,daemon=True);self.worker.start()

    def receive(self,rgb,depth):
        with self.lock:self.packet=(rgb,depth)

    def process(self):
        while not self.stop.wait(.25):
            with self.lock:packet=self.packet;self.packet=None
            # Mapping cannot act on callers, and memory accepts observations
            # only after ready. Avoid competing with SLAM/physics for CPU then.
            if not read_json(DATA/'ready.json',{}).get('ready'):continue
            # Once a face has been confirmed the safety loop needs fresh face
            # frames, not repeated appearance identification of the same body.
            if read_json(DATA/'mission-state.json',{}).get('phase') in (
                'waiting_for_raised_hand','planning','plan_acceptance','plan_result',
                'sending_navigation','walking','checking_stop','face_recovery_stop',
                'face_view_acceptance','face_view_walking','cancelling'):continue
            if not self.warmup.done():continue
            if packet is None or self.info is None:continue
            rgb,depth=packet;stamp=Time.from_msg(rgb.header.stamp)
            if not 0<=(self.get_clock().now()-stamp).nanoseconds/1e9<1.2:continue
            try:
                image=self.bridge.imgmsg_to_cv2(rgb,'bgr8');z=self.bridge.imgmsg_to_cv2(depth,'32FC1')
                rows,latency=self.pool.submit(infer_body_frame,image).result(timeout=10)
                tr=self.tf.lookup_transform('map',rgb.header.frame_id,stamp,timeout=Duration(seconds=.3)).transform
                q=tr.rotation;rot=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
                translation=np.array([tr.translation.x,tr.translation.y,tr.translation.z])
                msg=PersonTracks();msg.header.stamp=rgb.header.stamp;msg.header.frame_id='map'
                for row in rows:
                    row.pop('embedding',None)
                    row.pop('colour',None)
                    k=self.info.k
                    row['measured_attributes']=pose_attributes(row['keypoints'],z,(k[0],k[4],k[2],k[5]))
                    identity=row['identity']
                    if identity not in self.ids:continue
                    u,v=np.rint(torso_pixel(row)).astype(int);patch=z[max(0,v-4):v+5,max(0,u-4):u+5]
                    valid=patch[np.isfinite(patch)&(patch>.4)&(patch<9)]
                    if len(valid)<20:continue
                    distance=float(np.median(valid));k=self.info.k
                    camera=np.array([(u-k[2])*distance/k[0],(v-k[5])*distance/k[4],distance]);p=rot@camera+translation
                    row['position']=p.tolist()
                    msg.people.append(PersonTrack(track_id=self.ids.index(identity)+1,confidence=float(row['score']),raised_hand=False,
                        position=Point(x=float(p[0]),y=float(p[1]),z=float(p[2])),range_m=float(np.linalg.norm(camera)),bbox=list(map(float,row['box']))))
                age=(self.get_clock().now()-stamp).nanoseconds/1e9
                if age<=1.2:self.pub.publish(msg)
                write_json(DATA/'body-observations.json',{'stamp':stamp.nanoseconds/1e9,'latency_s':latency,'age_s':age,
                    'bodies':rows,'input':'YOLO pose ROI + pretrained OSNet + aligned depth + measurement-time TF',
                    'final_identity_confirmed':False})
                for row in rows:
                    x0,y0,x1,y1=row['box'];colour=(50,200,80) if row['identity'] else (80,150,230)
                    cv2.rectangle(image,(x0,y0),(x1,y1),colour,2)
                    cv2.putText(image,(row['identity'] or 'unknown')+f" {row['score']:.2f} body hint",
                        (x0,max(18,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,colour,1)
                ok,jpeg=cv2.imencode('.jpg',image)
                if ok:atomic(DATA/'body.jpg',jpeg.tobytes())
            except (ValueError,TransformException) as exc:self.get_logger().warning(str(exc),throttle_duration_sec=5)
            except Exception as exc:self.get_logger().error(str(exc),throttle_duration_sec=5)


def main():
    rclpy.init();node=BodyNode()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.stop.set();node.worker.join(timeout=3);node.pool.shutdown(wait=False,cancel_futures=True)
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
