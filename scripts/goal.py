#!/usr/bin/env python3
"""Use the public Nav2 actions to plan and execute a goal in the observed map."""
import argparse
import json
import math
import time
from pathlib import Path
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import Odometry, Path as NavPath
from action_msgs.msg import GoalStatus
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from tf2_ros import Buffer, TransformListener


def main():
    p = argparse.ArgumentParser()
    p.add_argument('x', type=float)
    p.add_argument('y', type=float)
    p.add_argument('--yaw', type=float, default=0)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--timeout', type=float, default=240)
    p.add_argument('--output', default='/workspace/logs/navigation.json')
    p.add_argument('--odom-topic', default='/odom')
    p.add_argument('--position-tolerance', type=float, default=.35)
    args = p.parse_args()
    rclpy.init()
    node = Node('navigation_test')
    tf = Buffer()
    listener = TransformListener(tf, node)
    odometry = []
    sub = node.create_subscription(Odometry, args.odom_topic, lambda m: odometry.append((m.pose.pose.position.x, m.pose.pose.position.y)), 10)
    observed_paths = []
    plan_sub = node.create_subscription(NavPath, '/plan',
        lambda m: observed_paths.append(m) if len(m.poses) > 1 and not observed_paths else None, 10)
    deadline = time.monotonic() + args.timeout
    def wait(future, until=None):
        until = deadline if until is None else until
        while not future.done() and time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert future.done(), 'Nav2 action timed out'
        return future.result()
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x, pose.pose.position.y = args.x, args.y
    pose.pose.orientation.z = math.sin(args.yaw / 2)
    pose.pose.orientation.w = math.cos(args.yaw / 2)
    # An action endpoint exists even while its lifecycle server is inactive.
    # Wait for Nav2's actual readiness rather than sending a rejected goal.
    for name in ('planner_server',) if args.plan_only else ('planner_server', 'bt_navigator', 'collision_monitor'):
        client = node.create_client(GetState, '/' + name + '/get_state')
        assert client.wait_for_service(timeout_sec=60), f'{name} lifecycle unavailable'
        while time.monotonic() < deadline:
            if wait(client.call_async(GetState.Request())).current_state.id == State.PRIMARY_STATE_ACTIVE:
                break
            time.sleep(.2)
        else:
            raise TimeoutError(name + ' did not activate')
        node.destroy_client(client)
    report = {'goal': [args.x, args.y, args.yaw], 'execution': 'not requested'}
    if args.plan_only:
        planner = ActionClient(node, ComputePathToPose, '/compute_path_to_pose')
        assert planner.wait_for_server(timeout_sec=90), 'Planner unavailable'
        goal = ComputePathToPose.Goal()
        goal.goal = pose
        goal.planner_id = 'GridBased'
        handle = wait(planner.send_goal_async(goal))
        assert handle.accepted, 'Plan rejected'
        result = wait(handle.get_result_async())
        assert result.status == GoalStatus.STATUS_SUCCEEDED, f'Planning failed: {result.result}'
        path = result.result.path.poses
    if not args.plan_only:
        # NavigateToPose owns planning and its standard costmap recovery tree.
        # A separate mandatory ComputePath preflight bypasses that recovery.
        navigator = ActionClient(node, NavigateToPose, '/navigate_to_pose')
        assert navigator.wait_for_server(timeout_sec=60), 'Navigator unavailable'
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = pose
        handle = wait(navigator.send_goal_async(nav_goal))
        assert handle.accepted, 'Navigation rejected'
        future = handle.get_result_async()
        try:
            result = wait(future)
        except BaseException:
            wait(handle.cancel_goal_async(), until=time.monotonic() + 10)
            raise
        assert result.status == GoalStatus.STATUS_SUCCEEDED, f'Navigation failed: {result.result}'
        end = tf.lookup_transform('map', 'base_footprint', Time()).transform.translation
        error = math.hypot(end.x-args.x, end.y-args.y)
        assert error < args.position_tolerance, f'Goal position error: {error}'
        traveled = sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(odometry,odometry[1:]))
        assert traveled > 0.1, 'No odometry motion observed'
        report.update(execution='succeeded', final_map_xy=[end.x,end.y], goal_error_m=error,
                      odom_topic=args.odom_topic, odom_distance_m=traveled)
        assert observed_paths, 'No Nav2 path observed during execution'
        path = observed_paths[0].poses
    assert len(path) > 1, 'Empty path'
    length = sum(math.hypot(b.pose.position.x-a.pose.position.x, b.pose.position.y-a.pose.position.y) for a,b in zip(path,path[1:]))
    report.update(path_poses=len(path), path_length_m=length,
                  plan=[[p.pose.position.x, p.pose.position.y] for p in path])
    report['passed'] = True
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='plan'}, indent=2))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
