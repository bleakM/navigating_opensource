#!/usr/bin/env python3
"""Gate Nav2 startup on live estimated odometry and the first observed SLAM map."""
import time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from rclpy.qos import QoSProfile,DurabilityPolicy
from nav_msgs.msg import Odometry,OccupancyGrid
from tf2_ros import Buffer,TransformListener


def main():
    rclpy.init();node=Node('workbench_mapping_startup',
        parameter_overrides=[Parameter('use_sim_time',value=True)])
    buffer=Buffer();listener=TransformListener(buffer,node);samples=[];observed=[False]
    node.create_subscription(Odometry,'/g1_odometry_publisher/odom',
        lambda m:samples.append(m.header.stamp.sec+m.header.stamp.nanosec/1e9),10)
    node.create_subscription(OccupancyGrid,'/map',lambda m:observed.__setitem__(0,any(v==0 for v in m.data)),
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    ready=False;end=time.monotonic()+120
    try:
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.1)
            now=node.get_clock().now().nanoseconds/1e9
            if (len(samples)>=20 and 0<=now-samples[-1]<.5 and observed[0]
                    and buffer.can_transform('map','base_footprint',Time())):
                ready=True;break
        node.get_logger().info('Estimated odometry and observed SLAM map ready' if ready else
            'Mapping sensors not ready; Nav2 remains disabled')
    finally:node.destroy_node();rclpy.try_shutdown()
    return 0 if ready else 1

if __name__=='__main__':raise SystemExit(main())
