#!/usr/bin/env python3
"""Start-up gate only: wait for RTAB-Map's real map match before starting Nav2."""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rtabmap_msgs.msg import Info
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener


def main():
    rclpy.init()
    node = Node('g1_localization_startup_gate', parameter_overrides=[
        rclpy.parameter.Parameter('use_sim_time', value=True)])
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    matches = []
    clouds = []

    def info(msg):
        if msg.loop_closure_id > 0 or msg.proximity_detection_id > 0:
            matches.append(time.monotonic())

    subscription = node.create_subscription(Info, '/rtabmap/info', info, 10)
    cloud_subscription = node.create_subscription(PointCloud2, '/rtabmap/cloud_map',
        lambda msg: clouds.append(msg.width * msg.height),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    deadline = time.monotonic() + 180
    ready = False
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if (len(matches) >= 2 and time.monotonic()-matches[0] > 3
                    and time.monotonic()-matches[-1] < 5
                    and clouds and clouds[-1] > 1000
                    and buffer.can_transform('map', 'base_footprint', rclpy.time.Time())):
                ready = True
                break
        if ready:
            node.get_logger().info('RTAB-Map matched and published the saved 3D map; Nav2 may now start.')
        if not ready:
            node.get_logger().error('No settled RTAB-Map match within 180 seconds; keeping Nav2 off.')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
