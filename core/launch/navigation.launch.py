"""Reuse baseline Nav2 launch with an interaction-only parameter file."""
import importlib.util
from launch.actions import TimerAction, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    spec=importlib.util.spec_from_file_location('baseline_navigation','/workspace/g1/launch/navigation.launch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.IncludeLaunchDescription
    def include(*args,**kwargs):
        arguments=dict(kwargs.get('launch_arguments',[]))
        if 'params_file' in arguments:
            arguments['params_file']='/workspace/core/config/nav2.yaml'
            kwargs['launch_arguments']=arguments.items()
            # Upstream appends its BT paths after params_file. Adapt that last
            # dictionary as well, otherwise our configured tree never runs.
            args=(PythonLaunchDescriptionSource('/workspace/core/launch/nav2_servers.launch.py'),*args[1:])
        return original(*args,**kwargs)
    module.IncludeLaunchDescription=include
    description=module.generate_launch_description()
    map_node=Node(package='nav2_costmap_2d',executable='nav2_costmap_2d',
        namespace='navigation_map',name='costmap',output='screen',
        parameters=['/workspace/core/config/navigation-map.yaml'])
    description.add_action(map_node)
    description.add_action(RegisterEventHandler(OnProcessExit(target_action=map_node,
        on_exit=[EmitEvent(event=Shutdown(reason='Required navigation map process exited'))])))
    description.add_action(TimerAction(period=3.,actions=[Node(package='nav2_lifecycle_manager',
        executable='lifecycle_manager',name='lifecycle_manager_navigation_map',output='screen',
        # The standalone Costmap2DROS executable is a lifecycle follower and
        # does not create a bond. Main Nav2 servers retain their normal bonds;
        # the map is checked by ready_ros and its process exit shuts us down.
        parameters=[{'use_sim_time':True,'autostart':True,'bond_timeout':0.0,
                     'node_names':['navigation_map/costmap']}])]))
    return description
