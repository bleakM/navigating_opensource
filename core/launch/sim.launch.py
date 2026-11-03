"""Separate simulator executable and actor-aware sensor visibility; original files untouched."""
import importlib.util
from pathlib import Path
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share=Path(get_package_share_directory('g1_bringup'))
    spec=importlib.util.spec_from_file_location('workbench_sim',share/'launch/sim.launch.py');sim=importlib.util.module_from_spec(spec);spec.loader.exec_module(sim)
    sim.UNITREE_MUJOCO_BIN='/workspace/.runtime/core-native/build/unitree_mujoco'
    sim.STAGED_SCENE_NAME=str(Path(sim.G1_MODEL_DIR)/sim.STAGED_SCENE_NAME)
    # Absolute scene avoids the executable-relative upstream unitree_robots path.
    old_simulator=sim._simulator
    def simulator(env,headless,delay):
        env['XIAOBEI_WORKBENCH']='/workspace/.runtime/session'
        return old_simulator(env,headless,delay)
    sim._simulator=simulator
    sim.CAMERA_MOUNTS='/workspace/core/config/cameras.yaml'
    original=sim._stage_scene
    def stage(world,sensors,pin_pelvis):
        staged=original(world,sensors,pin_pelvis)
        sim._stage_file('/workspace/.runtime/session/scene.xml',str(Path(sim.G1_MODEL_DIR)/sim.STAGED_WALK_BASE))
        return staged
    sim._stage_scene=stage
    sensors=sim._sensor_nodes
    def sensor_nodes(env,rviz,world,cameras):
        result=sensors(env,rviz,world,cameras)
        env['GROVE_G1_SENSOR_CONFIG']='/workspace/interaction/config/sim_sensors.yaml'
        env['XIAOBEI_WORKBENCH']='/workspace/.runtime/session'
        return result
    sim._sensor_nodes=sensor_nodes
    node=sim.Node
    def factory(**kwargs):
        if kwargs.get('name')=='g1_sensor_relay':kwargs['parameters'].append('/workspace/interaction/config/relay.yaml')
        return node(**kwargs)
    sim.Node=factory
    return sim.generate_launch_description()
