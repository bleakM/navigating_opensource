#!/usr/bin/env python3
"""Export live SLAM products and a committed RTAB-Map database snapshot."""
import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from rtabmap_msgs.msg import MapGraph
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='indoor')
    parser.add_argument('--database', help='Active RTAB-Map database; defaults to the Gazebo run marker')
    parser.add_argument('--odom-topic', default='/odom')
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        parser.error('--name must be a filename stem')
    root = Path('/workspace')
    stem = root / 'maps' / args.name
    db = Path(args.database or (root / '.runtime/active_database').read_text().strip())
    if db == stem.with_suffix('.db'):
        raise RuntimeError('Choose another name when snapshotting a localization database')
    rclpy.init()
    node = Node('save_slam_products')
    data = {}
    velocity = [float('inf'), float('inf')]
    last_odometry = [0.0]
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    subscriptions = [
        node.create_subscription(PointCloud2, '/rtabmap/cloud_map', lambda m: data.update(cloud=m), qos),
        node.create_subscription(MapGraph, '/rtabmap/mapGraph', lambda m: data.update(graph=m), qos),
    ]
    def moving(msg):
        velocity[:] = [math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y),
                       msg.twist.twist.angular.z]
        last_odometry[0] = time.monotonic()
    subscriptions.append(node.create_subscription(Odometry, args.odom_topic, moving, 10))
    stop_explore = node.create_publisher(Bool, '/explore/resume', 10)
    def call(name):
        client = node.create_client(Empty, '/rtabmap/rtabmap/' + name)
        if not client.wait_for_service(timeout_sec=10):
            raise RuntimeError('Missing RTAB-Map service: ' + name)
        future = client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=90)
        if not future.done() or future.exception():
            raise RuntimeError('RTAB-Map service failed: ' + name)
        node.destroy_client(client)
    try:
        end = time.monotonic() + 45
        while len(data) < 2 and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
        assert len(data) == 2, 'Missing latched cloud/mapGraph'
        # The official backup service reinitializes graph memory. Stop exploration first,
        # and keep SLAM paused afterwards; restart with localize to continue navigation.
        end = time.monotonic() + 20
        quiet_since = None
        while time.monotonic() < end:
            stop_explore.publish(Bool(data=False))
            rclpy.spin_once(node, timeout_sec=0.1)
            quiet = max(map(abs, velocity)) < 0.01 and time.monotonic() - last_odometry[0] < 1.0
            quiet_since = (quiet_since or time.monotonic()) if quiet else None
            if quiet_since and time.monotonic()-quiet_since > 2:
                break
        else:
            raise RuntimeError('Robot has not stopped; cancel navigation before saving')
        call('pause')
        try:
            cloud = data['cloud']
            points = point_cloud2.read_points(cloud, skip_nans=True)
            points = points[np.isfinite(points['x']) & np.isfinite(points['y']) & np.isfinite(points['z'])]
            span = {axis: float(np.ptp(points[axis])) for axis in ('x', 'y', 'z')}
            assert min(span['x'], span['y']) > 1. and span['z'] > .5, 'Cannot save a flattened or incomplete 3D map'
            # An old backup may share an immutable saved-map inode. Set it aside
            # before the official callback creates a fresh backup; never let a
            # later save overwrite a previously archived snapshot in place.
            source = Path(str(db) + '.back')
            if source.exists():
                source.rename(Path(str(source) + '.saved-' + str(time.time_ns())))
            call('backup')  # Official callback flushes SQLite and creates .db.back.
            source = Path(str(db) + '.back')
            assert source.is_file() and source.stat().st_size > 0
            temp = stem.with_suffix('.db.partial')
            assert not temp.exists(), 'An unfinished map snapshot already exists'
            os.link(source, temp)
            with sqlite3.connect(f'file:{temp}?mode=ro', uri=True) as conn:
                assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            temp.replace(stem.with_suffix('.db'))
            stem.with_suffix('.db').chmod(0o444)
            subprocess.run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', str(stem),
                            '--ros-args', '-p', 'use_sim_time:=true', '-p', 'save_map_timeout:=30.0'],
                           check=True, timeout=45)
            colored = 'rgb' in points.dtype.names
            dtype = [('x','<f4'), ('y','<f4'), ('z','<f4')]
            if colored:
                dtype += [('red','u1'), ('green','u1'), ('blue','u1')]
            output = np.empty(len(points), dtype=dtype)
            for field in ('x','y','z'):
                output[field] = points[field]
            if colored:
                rgb = np.ascontiguousarray(points['rgb']).view(np.uint32)
                for name, shift in [('red',16), ('green',8), ('blue',0)]:
                    output[name] = (rgb >> shift) & 255
            header = f'ply\nformat binary_little_endian 1.0\nelement vertex {len(output)}\n'
            header += 'property float x\nproperty float y\nproperty float z\n'
            if colored:
                header += 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            with stem.with_suffix('.ply').open('wb') as stream:
                stream.write((header + 'end_header\n').encode())
                stream.write(output.tobytes())
            graph = data['graph']
            nodes = [{'id': i, 'xyz': [p.position.x,p.position.y,p.position.z],
                      'xyzw': [p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]}
                     for i,p in zip(graph.poses_id,graph.poses)]
            stem.with_suffix('.graph.json').write_text(json.dumps(nodes,indent=2)+'\n')
            report = {'source_database': str(db), 'cloud_frame': cloud.header.frame_id,
                      'cloud_xyz_span_m': span,
                      'cloud_points': len(output), 'graph_nodes': len(nodes),
                      'database_sha256': hashlib.file_digest(stem.with_suffix('.db').open('rb'),'sha256').hexdigest(),
                      'sqlite_integrity': 'ok'}
            stem.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2), flush=True)
        finally:
            print('SLAM remains paused after backup. Restart scripts/run localize to continue.',flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
