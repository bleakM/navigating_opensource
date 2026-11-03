"""Reuse frozen RTAB-Map launch with workbench-only occupancy height configuration."""
import importlib.util
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    spec=importlib.util.spec_from_file_location('baseline_slam','/workspace/g1/launch/slam.launch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.Node
    def node(**kwargs):
        kwargs['parameters']=[('/workspace/core/config/rtabmap.yaml' if p=='/workspace/g1/config/rtabmap.yaml' else p) for p in kwargs.get('parameters',[])]
        kwargs['parameters'].append({'initial_pose':LaunchConfiguration('initial_pose')})
        return original(**kwargs)
    module.Node=node
    description=module.generate_launch_description()
    return LaunchDescription([DeclareLaunchArgument('initial_pose',default_value=''),*description.entities])
