"""Simulator-side scene staging; actor group4 is non-colliding and absent from mapping."""
import json,math,os,xml.etree.ElementTree as ET
from pathlib import Path
from common import ROOT,DATA,write_json


def build_scene(selected,resource,output):
    source=Path(resource['scene'])
    text=source.read_text().replace('@GROVE_ASSETS@',os.environ.get('GROVE_ASSETS_DIR',str(ROOT/'third_party/grove-g1/workspace/assets')))
    root=ET.fromstring(text)
    asset=root.find('asset');world=root.find('worldbody')
    if asset is None:asset=ET.SubElement(root,'asset')
    for b in list(world.findall('body')):
        if b.get('name','').startswith('interaction_person'):world.remove(b)
    for a in list(asset):
        if a.get('name','').startswith('interaction_'):asset.remove(a)
    # Resolve the robot include independently of the render process's current directory.
    modeldir=Path('/opt/unitree_robotics/unitree_mujoco/unitree_robots/g1')
    robot_file=modeldir/'g1_29dof.xml'
    spawn=resource.get('spawn',[0.,0.])
    if any(abs(float(v))>1e-6 for v in spawn):
        robot_root=ET.parse(robot_file).getroot();pelvis=robot_root.find("worldbody/body[@name='pelvis']")
        if pelvis is None:raise ValueError('G1模型缺少pelvis起点')
        position=[float(v) for v in pelvis.get('pos','0 0 .8').split()];position[:2]=spawn
        pelvis.set('pos',' '.join(str(v) for v in position))
        robot_root.find('compiler').set('meshdir',str(modeldir/'meshes'))
        robot_file=Path(output).parent/'spawn-robot.xml';ET.ElementTree(robot_root).write(robot_file,encoding='unicode')
    for include in root.findall('include'):
        if include.get('file')=='g1_29dof.xml':include.set('file',str(robot_file))
    for element in root.iter():
        filename=element.get('file')
        if filename and not Path(filename).is_absolute():element.set('file',str((source.parent/filename).resolve()))
    # Included G1 meshdir is relative to the top-level MJCF, not its own include.
    compiler=root.find('compiler')
    if compiler is None:compiler=ET.SubElement(root,'compiler')
    compiler.set('meshdir',str(modeldir/'meshes'))
    if not any(c.get('name')=='interaction_rgbd' for c in world.findall('camera')):
        ET.SubElement(world,'camera',name='interaction_rgbd',pos='0 0 1.2',fovy='45',resolution='640 480')
    for i,person in enumerate(selected):
        name=person['id'];folder=ROOT/'assets/avatars'/name
        ET.SubElement(asset,'mesh',name=f'wb_mesh_{i}',file=str(folder/'person-raised.obj'))
        ET.SubElement(asset,'texture',name=f'wb_texture_{i}',type='2d',file=str(folder/'person.png'))
        ET.SubElement(asset,'material',name=f'wb_material_{i}',texture=f'wb_texture_{i}',specular='.05')
        body=ET.SubElement(world,'body',name=f'wb_actor_{i}',mocap='true',pos=f"{person['x']} {person['y']} 0",euler='0 0 -1.5707963267948966')
        ET.SubElement(body,'geom',type='mesh',mesh=f'wb_mesh_{i}',material=f'wb_material_{i}',contype='0',conaffinity='0',group='4',mass='0')
    # All authored paths passed to C++ use its /workspace mount.
    ET.indent(root)
    Path(output).write_text(ET.tostring(root,encoding='unicode'))
    return output
