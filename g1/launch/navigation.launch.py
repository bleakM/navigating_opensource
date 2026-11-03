"""Grove's tested G1 Nav2 servers. RTAB-Map is the sole global TF authority."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    share=Path(get_package_share_directory('g1_navigation'))
    def include(name,**args):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share/'launch'/name)),launch_arguments=args.items())
    return LaunchDescription([
        SetParameter(name='use_sim_time',value=True),
        # Reuse RTAB-Map's standard depth converter, including decimation and voxel filtering.
        # Low tables can be below the Mid360's nearby vertical field of view.
        Node(package='rtabmap_util', executable='point_cloud_xyz', name='chest_depth_points',
             output='screen', parameters=[{'use_sim_time': True, 'decimation': 4,
                 'voxel_size': 0.05, 'min_depth': 0.4, 'max_depth': 5.0, 'qos': 2,
                 'qos_camera_info': 2, 'filter_nans': True}],
             remappings=[('depth/image', '/chest_camera/aligned_depth_to_color/image_raw'),
                         ('depth/camera_info', '/chest_camera/aligned_depth_to_color/camera_info'),
                         ('cloud', '/chest_camera/points')]),
        include('scan.launch.py',use_composition='false'),
        include('nav2.launch.py',use_sim_time='true',params_file='/workspace/g1/config/nav2.yaml',keepout='false'),
    ])
