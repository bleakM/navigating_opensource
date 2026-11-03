"""MuJoCo-only core sensor/SLAM/Nav2 session, no periodic synthetic callers."""
from pathlib import Path
import os,json,time,hashlib
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,ExecuteProcess,IncludeLaunchDescription,GroupAction,OpaqueFunction,RegisterEventHandler,EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node,SetParameter
from ament_index_python.packages import get_package_share_directory


def start(context):
    mode=LaunchConfiguration('mode').perform(context);database=LaunchConfiguration('database').perform(context)
    resource=json.loads(Path('/workspace/.runtime/session/resource.json').read_text())
    # Map identity only. No scene geometry or actor authoring enters memory.
    saved=Path(resource.get('saved',''))
    if mode=='saved' and saved.is_file():
        stat=saved.stat();key=f"{resource['id']}:{saved}:{stat.st_size}:{stat.st_mtime_ns}"
    else:key=f"{resource['id']}:fresh:{time.time_ns()}"
    memory_scope=hashlib.sha256(key.encode()).hexdigest()[:24]
    def include(path,**args):return GroupAction([SetParameter(name='use_sim_time',value=True),IncludeLaunchDescription(PythonLaunchDescriptionSource(path),launch_arguments=args.items())])
    python='/workspace/.runtime/python/bin/python'
    env={'PYTHONPATH':'/workspace/core/src:/workspace/interaction/src:'+os.environ.get('PYTHONPATH',''),'YOLO_OFFLINE':'true','YOLO_AUTOINSTALL':'false'}
    def process(name,*args):return ExecuteProcess(cmd=[python,'/workspace/core/src/'+name+'.py',*args],additional_env=env,output='screen')
    actions=[SetParameter(name='use_sim_time',value=True),SetParameter(name='publish_sim_clock',value=False),SetParameter(name='stamp_with_sim_time',value=True),
        include('/workspace/core/launch/sim.launch.py',sensors='true',odometry='fast_lio',pin_pelvis='false',rviz='false',world='apartment',headless=str(os.environ.get('WB_HARDWARE_GL')!='1').lower(),cameras='chest,interaction',controllers_overrides='/workspace/g1/config/sim-clock.yaml'),
        include('/workspace/core/launch/slam.launch.py',database=database,localization=str(mode=='saved').lower(),initial_pose=LaunchConfiguration('initial_pose').perform(context)),
        Node(package='tf2_ros',executable='static_transform_publisher',name='workbench_camera_mount',arguments=['--x','.13','--y','0','--z','.40','--pitch','-0.17453292519943295','--frame-id','torso_link','--child-frame-id','interaction_camera_link']),
        Node(package='tf2_ros',executable='static_transform_publisher',name='workbench_camera_optical',arguments=['--roll','-1.57079632679','--pitch','0','--yaw','-1.57079632679','--frame-id','interaction_camera_link','--child-frame-id','interaction_camera_color_optical_frame']),
        Node(package='tf2_ros',executable='static_transform_publisher',name='workbench_microphone_mount',arguments=['--x','0','--y','0','--z','1.35','--frame-id','base_footprint','--child-frame-id','microphone_array']),
        include(str(Path(get_package_share_directory('canopy'))/'launch/world_model.launch.py'),cameras='chest=/chest_camera',odom_topic='/g1_odometry_publisher/odom',cloud_topic='/livox/lidar',map_topic='/navigation_map/costmap',params_file='/workspace/core/config/canopy.yaml',world_dir='',detector='false',describe='false',rviz='false'),
        process('faces_ros'),process('audio_ros','simulate'),process('audio_ros','perceive'),process('telemetry'),process('mission','--ros-args','-r','/map:=/navigation_map/costmap','-p','memory_scope:='+memory_scope),
        ExecuteProcess(cmd=[python,'/workspace/core/src/arms.py','--report','/workspace/.runtime/session/arms.json'],additional_env=env,output='screen')]
    resources=ExecuteProcess(cmd=['python3','/workspace/scripts/g1_resources.py','--launch-pid',str(os.getpid()),'--report','/workspace/.runtime/session/resources-health.json'],output='screen')
    def resources_exit(event,ctx):
        if ctx.is_shutdown or event.returncode==0:return []
        Path('/workspace/.runtime/session/ready.json').write_text('{"ready":false,"phase":"failed","error":"资源监督停止了仿真，请查看 resources-health.json"}')
        return [EmitEvent(event=Shutdown(reason='Resource health check failed'))]
    actions += [RegisterEventHandler(OnProcessExit(target_action=resources,on_exit=resources_exit)),resources]
    if mode=='saved':
        gate=ExecuteProcess(cmd=['python3','/workspace/scripts/wait_g1_localization.py'],output='screen')
        def after(event,ctx):
            if ctx.is_shutdown:return []
            if event.returncode:
                Path('/workspace/.runtime/session/ready.json').write_text('{"ready":false,"phase":"failed","error":"RTAB-Map未在期限内完成稳定地图匹配，未启用导航与呼叫"}')
                return [EmitEvent(event=Shutdown(reason='Map matching failed'))]
            Path('/workspace/.runtime/session/ready.json').write_text('{"ready":false,"phase":"navigation_starting","source":"actual saved map matching passed; waiting Nav2"}')
            return [include('/workspace/core/launch/navigation.launch.py'),process('ready_ros')]
        actions += [RegisterEventHandler(OnProcessExit(target_action=gate,on_exit=after)),gate]
    else:
        gate=process('mapping_startup')
        def after_mapping_sensors(event,ctx):
            if ctx.is_shutdown:return []
            if event.returncode:
                Path('/workspace/.runtime/session/ready.json').write_text('{"ready":false,"phase":"failed","error":"建图传感器与里程计启动超时，未启用导航"}')
                return [EmitEvent(event=Shutdown(reason='Mapping sensor startup gate failed'))]
            return [include('/workspace/core/launch/navigation.launch.py'),process('explore')]
        actions += [RegisterEventHandler(OnProcessExit(target_action=gate,on_exit=after_mapping_sensors)),gate]
    return actions


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('mode',default_value='fresh'),DeclareLaunchArgument('database'),DeclareLaunchArgument('initial_pose',default_value=''),OpaqueFunction(function=start)])
