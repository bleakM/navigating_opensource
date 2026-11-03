#!/usr/bin/env python3
"""Observed cloud/grid/path visualization; no ground truth is subscribed here."""
import time
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile,DurabilityPolicy,qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import OccupancyGrid,Path,Odometry
from tf2_ros import Buffer,TransformListener,TransformException
from rclpy.time import Time
from common import DATA,write_json


class Telemetry(Node):
    def __init__(self):
        super().__init__('workbench_observed_map',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.last_cloud=0
        self.previous_odom=None;self.distance=0.;self.free_area=0.;self.started=time.monotonic()
        self.tf=Buffer();self.listener=TransformListener(self.tf,self)
        self.create_timer(.2,self.pose)
        self.create_subscription(PointCloud2,'/rtabmap/cloud_map',self.cloud,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(OccupancyGrid,'/map',self.grid,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Path,'/plan',self.path,10)
        self.create_subscription(Odometry,'/g1_odometry_publisher/odom',self.odometry,10)
        self.create_timer(1.,self.mapping_metrics)

    def odometry(self,msg):
        p=msg.pose.pose.position;xy=np.asarray([p.x,p.y])
        if not np.isfinite(xy).all():return
        if self.previous_odom is not None:self.distance+=float(np.linalg.norm(xy-self.previous_odom))
        self.previous_odom=xy

    def mapping_metrics(self):
        write_json(DATA/'mapping-telemetry.json',{'observed_free_area_m2':self.free_area,
            'odom_distance_m':self.distance,'wall_duration_s':time.monotonic()-self.started,
            'metrics_source':'current /map known-free cells and FAST-LIO2 odometry XY integral; no truth'})

    def cloud(self,msg):
        if time.monotonic()-self.last_cloud<.75:return
        self.last_cloud=time.monotonic()
        fields={f.name for f in msg.fields};color='rgb' if 'rgb' in fields else 'rgba' if 'rgba' in fields else None
        names=['x','y','z']+([color] if color else [])
        points=point_cloud2.read_points(msg,field_names=names,skip_nans=True)
        stride=max(1,(len(points)+17999)//18000);points=points[::stride]
        xyz=np.column_stack([points[n] for n in ['x','y','z']]);valid=np.isfinite(xyz).all(axis=1);xyz=xyz[valid]
        colors=None
        if color:
            packed=np.ascontiguousarray(points[color][valid])
            if packed.dtype.kind=='f':packed=packed.view(np.uint32)
            else:packed=packed.astype(np.uint32)
            colors=np.column_stack([(packed>>16)&255,(packed>>8)&255,packed&255]).tolist()
        write_json(DATA/'cloud.json',{'frame':msg.header.frame_id,'stamp':msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,'received_wall':time.time(),'original_points':msg.width*msg.height,'points':np.round(xyz,3).tolist(),'colors':colors,'color_source':color,'source':'/rtabmap/cloud_map'})

    def pose(self):
        try:
            t=self.tf.lookup_transform('map','base_footprint',Time())
            stamp=t.header.stamp.sec+t.header.stamp.nanosec/1e9
            if not 0<=(self.get_clock().now().nanoseconds/1e9-stamp)<1.5:return
            p=t.transform.translation;q=t.transform.rotation
            write_json(DATA/'localization.json',{'frame':'map','stamp':t.header.stamp.sec+t.header.stamp.nanosec/1e9,'received_wall':time.time(),'position':[p.x,p.y,p.z],'quaternion':[q.x,q.y,q.z,q.w],'source':'TF map → base_footprint (FAST-LIO2 + RTAB-Map)'})
        except TransformException:pass

    def grid(self,msg):
        self.free_area=sum(v==0 for v in msg.data)*msg.info.resolution**2
        write_json(DATA/'observed-map.json',{'width':msg.info.width,'height':msg.info.height,'resolution':msg.info.resolution,'origin':[msg.info.origin.position.x,msg.info.origin.position.y],'data':list(msg.data),'source':'/map'})

    def path(self,msg):
        write_json(DATA/'path.json',{'poses':[[p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in msg.poses],'frame':msg.header.frame_id,'stamp':msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,'received_wall':time.time(),'source':'/plan (Nav2)'})


def main():
    rclpy.init();node=Telemetry()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
