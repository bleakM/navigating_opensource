#!/usr/bin/env python3
"""Face-only recognition and aligned-depth geometry from the actual robot camera."""
import threading,time
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
from identity import Faces
from common import DATA,ROOT,read_json,write_json,atomic


class FaceNode(Node):
    def __init__(self):
        super().__init__('xiaobei_face_recognition',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.faces=Faces();self.bridge=CvBridge();self.tf=Buffer();self.listener=TransformListener(self.tf,self)
        self.info=None;self.packet=None;self.lock=threading.Lock();self.stop=threading.Event()
        self.gallery=read_json(DATA/'face-gallery.json',{})
        self.ids=list(self.gallery)
        self.pub=self.create_publisher(PersonTracks,'/wb/faces',10)
        self.create_subscription(CameraInfo,'/interaction_camera/color/camera_info',lambda m:setattr(self,'info',m),qos_profile_sensor_data)
        rgb=message_filters.Subscriber(self,Image,'/interaction_camera/color/image_raw',qos_profile=qos_profile_sensor_data)
        depth=message_filters.Subscriber(self,Image,'/interaction_camera/aligned_depth_to_color/image_raw',qos_profile=qos_profile_sensor_data)
        self.sync=message_filters.ApproximateTimeSynchronizer([rgb,depth],3,.04);self.sync.registerCallback(self.receive)
        self.worker=threading.Thread(target=self.process,daemon=True);self.worker.start()

    def receive(self,rgb,depth):
        with self.lock:self.packet=(rgb,depth)

    def process(self):
        while not self.stop.wait(.22):
            with self.lock:packet=self.packet;self.packet=None
            if not packet or self.info is None:continue
            rgb,depth=packet;stamp=Time.from_msg(rgb.header.stamp)
            if not 0<=(self.get_clock().now()-stamp).nanoseconds/1e9<1.2:continue
            try:
                image=self.bridge.imgmsg_to_cv2(rgb,'bgr8');z=self.bridge.imgmsg_to_cv2(depth,'32FC1')
                detected=self.faces.identify(image,self.gallery)
                tr=self.tf.lookup_transform('map',rgb.header.frame_id,stamp,timeout=Duration(seconds=.85)).transform
                q=tr.rotation;rot=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix();translation=np.array([tr.translation.x,tr.translation.y,tr.translation.z])
                msg=PersonTracks();msg.header.stamp=rgb.header.stamp;msg.header.frame_id='map';rows=[]
                for face in detected:
                    x,y,w,h=face['box'];u=int(x+w*.5);v=int(y+h*.5)
                    patch=z[max(0,v-3):v+4,max(0,u-3):u+4];valid=patch[np.isfinite(patch)&(patch>.3)&(patch<9)]
                    identity=face['identity'];rows.append({'identity':identity,'score':face['score'],'box':face['box'],'scores':face['scores']})
                    cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+h)),(70,220,100) if identity else (100,150,240),2)
                    cv2.putText(image,(identity or 'unknown')+f" {face['score']:.2f}",(int(x),max(15,int(y)-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(70,220,100),1)
                    if identity is None or len(valid)<15:continue
                    distance=float(np.median(valid));k=self.info.k
                    camera=np.array([(u-k[2])*distance/k[0],(v-k[5])*distance/k[4],distance]);p=rot@camera+translation
                    msg.people.append(PersonTrack(track_id=self.ids.index(identity)+1,confidence=.9,raised_hand=False,position=Point(x=float(p[0]),y=float(p[1]),z=float(p[2])),range_m=float(np.linalg.norm(camera)),bbox=[float(x),float(y),float(x+w),float(y+h)]))
                if (self.get_clock().now()-stamp).nanoseconds/1e9<=1.2:self.pub.publish(msg)
                ok,jpeg=cv2.imencode('.jpg',image)
                if ok:atomic(DATA/'robot.jpg',jpeg.tobytes())
                write_json(DATA/'face-observations.json',{'stamp':stamp.nanoseconds/1e9,'faces':rows,'input':'YuNet face ROI + SFace only; no body/actor pose input'})
            except (ValueError,TransformException) as exc:self.get_logger().warning(str(exc),throttle_duration_sec=5)
            except Exception as exc:self.get_logger().error(str(exc),throttle_duration_sec=5)


def main():
    rclpy.init();node=FaceNode()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.stop.set();node.worker.join(timeout=3);node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
