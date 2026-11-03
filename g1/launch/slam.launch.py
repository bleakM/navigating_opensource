"""Shared ROS SLAM backend, fed exclusively by Grove's simulated sensors and FAST-LIO TF."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    slam = Node(package='rtabmap_slam',executable='rtabmap',name='rtabmap',namespace='rtabmap',
             output='screen',parameters=['/workspace/g1/config/rtabmap.yaml',{
                 'database_path':LaunchConfiguration('database'),
                 'Mem/IncrementalMemory':ParameterValue(PythonExpression(["'false' if '",LaunchConfiguration('localization'),"' == 'true' else 'true'"]),value_type=str),
                 'Mem/InitWMWithAllNodes':'true',
                 # The specified restart is the fixed map origin, rather than
                 # the place exploration ended. This is RTAB-Map's documented
                 # initialization policy, not a simulator pose measurement;
                 # the startup gate still requires actual map registrations.
                 'RGBD/StartAtOrigin':ParameterValue(PythonExpression(["'true' if '",LaunchConfiguration('localization'),"' == 'true' else 'false'"]),value_type=str),
                 # Mapping needs live-grid refresh for Canopy's observation
                 # acknowledgements. Localization keeps the saved grid fixed:
                 # an unmatched startup scan must never be fused at the saved
                 # last pose and leave a permanent obstacle ghost.
                 'map_always_update':ParameterValue(PythonExpression(["'",LaunchConfiguration('localization'),"' != 'true'"]),value_type=bool),
                 # Reassemble both outputs from stored local grids. Loading only
                 # the cached 2D map can mark nodes already assembled before a
                 # cloud subscriber exists, leaving the saved 3D cloud empty.
                 'use_saved_map':ParameterValue(PythonExpression(["'",LaunchConfiguration('localization'),"' != 'true'"]),value_type=bool),
                 # Localization must also process a stationary restart, rather
                 # than requiring a movement before another map match is tried.
                 'RGBD/LinearUpdate':ParameterValue(PythonExpression(["'0' if '",LaunchConfiguration('localization'),"' == 'true' else '0.2'"]),value_type=str),
                 'RGBD/AngularUpdate':ParameterValue(PythonExpression(["'0' if '",LaunchConfiguration('localization'),"' == 'true' else '0.15'"]),value_type=str),
             }],remappings=[('rgb/image','/chest_camera/color/image_raw'),
                            ('depth/image','/chest_camera/aligned_depth_to_color/image_raw'),
                            ('rgb/camera_info','/chest_camera/color/camera_info'),
                            ('scan_cloud','/livox/lidar'),('map','/map')])

    def slam_exited(event, context):
        if context.is_shutdown:
            return []
        return [EmitEvent(event=Shutdown(reason=f'Critical RTAB-Map process exited ({event.returncode})'))]

    return LaunchDescription([
        DeclareLaunchArgument('database', default_value='/workspace/maps/g1-session.db'),
        DeclareLaunchArgument('localization', default_value='false'),
        RegisterEventHandler(OnProcessExit(target_action=slam, on_exit=slam_exited)),
        slam,
    ])
