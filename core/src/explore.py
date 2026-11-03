#!/usr/bin/env python3
"""Supervise the existing Grove exploration tree, then save and enable calls."""
import signal,subprocess,time,sys
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from lifecycle_msgs.srv import GetState
from std_srvs.srv import Empty
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter as Param,ParameterValue,ParameterType
from ament_index_python.packages import get_package_prefix,get_package_share_directory
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile,DurabilityPolicy
from common import DATA,ROOT,write_json,read_json

LOCALIZATION_VALUES = {
    'Mem/IncrementalMemory': 'false',
    'RGBD/LinearUpdate': '0.0',
    'RGBD/AngularUpdate': '0.0',
}


def main():
    rclpy.init();node=Node('workbench_exploration',parameter_overrides=[Parameter('use_sim_time',value=True)])
    clients=[node.create_client(GetState,'/'+name+'/get_state') for name in ['planner_server','controller_server','smoother_server','velocity_smoother','bt_navigator','behavior_server','collision_monitor']]
    observed=[False]
    node.create_subscription(OccupancyGrid,'/map',lambda m:observed.__setitem__(0,any(v==0 for v in m.data)),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    child=None
    def wait(future,seconds=8):
        end=time.monotonic()+seconds
        while rclpy.ok() and not future.done() and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
        if not future.done():raise RuntimeError('ROS service timeout')
        return future.result()
    try:
        write_json(DATA/'ready.json',{'ready':False,'phase':'waiting_for_mapping_stack'})
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            if observed[0] and all(c.service_is_ready() for c in clients):
                if all(wait(c.call_async(GetState.Request())).current_state.id==3 for c in clients):break
            rclpy.spin_once(node,timeout_sec=.3)
        else:raise RuntimeError('Mapping startup timeout')
        # Original Grove frontier + coverage tree. Geometric coverage integrates
        # real depth images independently of the optional semantic detector.
        tree=ROOT/'core/config/trees/explore.xml'
        resource=read_json(DATA/'resource.json');map_name='workbench-'+resource['id']+'-'+str(time.time_ns())
        began=time.monotonic();sim_began=node.get_clock().now().nanoseconds/1e9
        command=[str(Path(get_package_prefix('g1_orchestration'))/'lib/g1_orchestration/g1_bt_executor'),'--ros-args','-p','use_sim_time:=true','-p','tree_file:='+str(tree),'-p','groot2_port:=0']
        child=subprocess.Popen(command)
        write_json(DATA/'ready.json',{'ready':False,'phase':'mapping'})
        while child.poll() is None:rclpy.spin_once(node,timeout_sec=.2)
        if child.returncode:raise RuntimeError('Grove exploration did not succeed')
        elapsed=time.monotonic()-began;sim_elapsed=node.get_clock().now().nanoseconds/1e9-sim_began
        write_json(DATA/'ready.json',{'ready':False,'phase':'saving'})
        child=subprocess.Popen(['python3','/workspace/scripts/save_map.py','--name',map_name,'--database',str(DATA/'mapping.db'),'--odom-topic','/g1_odometry_publisher/odom'])
        while child.poll() is None:rclpy.spin_once(node,timeout_sec=.2)
        if child.returncode:raise RuntimeError('Map save failed')
        write_json(DATA/'exploration.json',dict(**read_json(DATA/'mapping-telemetry.json',{}),status='saved',tree=str(tree),elapsed_wall_s=elapsed,elapsed_sim_s=sim_elapsed,tree_success=True,saved_database=str(ROOT/'maps'/(map_name+'.db')),coverage_audit='Not measured by the runtime; tree completion is not an independent coverage score'))
        client=node.create_client(SetParameters,'/rtabmap/rtabmap/set_parameters')
        # Match the frozen saved-map launch: even a stationary robot must keep
        # registering observations while the unchanged localization gate runs.
        values=LOCALIZATION_VALUES
        request=SetParameters.Request(parameters=[Param(name=k,value=ParameterValue(type=ParameterType.PARAMETER_STRING,string_value=v)) for k,v in values.items()])
        result=wait(client.call_async(request))
        if not all(r.successful for r in result.results):raise RuntimeError('Localization mode switch failed')
        resume=node.create_client(Empty,'/rtabmap/rtabmap/resume');wait(resume.call_async(Empty.Request()))
        write_json(DATA/'ready.json',{'ready':False,'phase':'localizing_saved_result'})
        child=subprocess.Popen(['python3','/workspace/scripts/wait_g1_localization.py'])
        while child.poll() is None:rclpy.spin_once(node,timeout_sec=.2)
        if child.returncode:raise RuntimeError('Saved mapping result failed actual localization gate')
        write_json(DATA/'ready.json',{'ready':True,'phase':'ready','resource':resource['id'],'saved_database':str(ROOT/'maps'/(map_name+'.db')),'saved_spawn':resource.get('spawn',[0.,0.]),'source':'Grove exploration tree SUCCESS + saved sensor map + actual localization gate','all_rooms_audited':False,'coverage_audit':'Not measured in source-only runtime'})
        while rclpy.ok():rclpy.spin_once(node,timeout_sec=.3)
    except KeyboardInterrupt:pass
    except Exception as exc:
        if rclpy.ok():write_json(DATA/'ready.json',{'ready':False,'phase':'failed','error':str(exc) or type(exc).__name__})
    finally:
        if child and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=8)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=5)
        node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
