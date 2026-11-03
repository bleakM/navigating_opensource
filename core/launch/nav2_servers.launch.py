"""Reuse Grove servers, with official Nav2 path and velocity smoothing."""
import importlib.util
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile


TREE='/workspace/core/config/search-navigation.xml'
PARAMS='/workspace/core/config/nav2.yaml'


def generate_launch_description():
    path=Path(get_package_share_directory('g1_navigation'))/'launch/nav2.launch.py'
    spec=importlib.util.spec_from_file_location('grove_workbench_nav2',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.Node
    def factory(**kwargs):
        name=kwargs.get('name')
        if name=='bt_navigator':
            kwargs['parameters']=[*kwargs['parameters'],{'default_nav_to_pose_bt_xml':TREE}]
        if name=='g1_base_approach':
            # Upstream StepClear supplies its own observed-costmap retreat
            # planner. Keep its commands in the same final safety chain.
            # Use an explicit node selector, matching the upstream YAML.
            # A generated /** dictionary loses to that exact node selector.
            kwargs['parameters']=[*kwargs['parameters'],
                '/workspace/core/config/base-approach.yaml']
            kwargs['remappings']=[*kwargs.get('remappings',[]),
                ('/cmd_vel','/cmd_vel_navigation')]
        if name in ('controller_server','behavior_server'):
            kwargs['remappings']=[(a,'/cmd_vel_navigation' if b=='/cmd_vel_raw' else b)
                                  for a,b in kwargs.get('remappings',[])]
        if name=='lifecycle_manager_navigation':
            kwargs['parameters']=[*kwargs['parameters'],{'node_names':[
                'controller_server','planner_server','smoother_server','velocity_smoother',
                'behavior_server','bt_navigator','collision_monitor']}]
        return original(**kwargs)
    module.Node=factory
    description=module.generate_launch_description()
    params=ParameterFile(PARAMS,allow_substs=True)
    description.add_action(Node(package='nav2_smoother',executable='smoother_server',
        name='smoother_server',parameters=[params],output='screen'))
    # Collision monitoring is AFTER smoothing, so a detected obstacle can stop
    # commands immediately instead of being softened by an acceleration ramp.
    description.add_action(Node(package='nav2_velocity_smoother',executable='velocity_smoother',
        name='velocity_smoother',parameters=[params],output='screen',
        remappings=[('cmd_vel','/cmd_vel_navigation'),('cmd_vel_smoothed','/cmd_vel_raw')]))
    return description
