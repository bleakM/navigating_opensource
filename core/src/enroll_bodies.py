#!/usr/bin/env python3
"""Multi-view asset registration, independent from the live simulation.

Synthetic registration is deliberate, like the existing face gallery. The
runtime only compares sensor crops to these templates; no actor IDs are read.
"""
import argparse,math
from xml.sax.saxutils import quoteattr
import cv2,mujoco,numpy as np
from body_identity import Bodies
from common import DATA,ROOT,read_json,write_json


def render_views(identity,height,angles=(0,45,-45,90,-90,180),distance=3.,background=False):
    folder=ROOT/'assets/avatars'/identity
    mesh=quoteattr(str(folder/'person-raised.obj'));texture=quoteattr(str(folder/'person.png'))
    xml=f'''<mujoco><visual><global offwidth="640" offheight="640"/>
      <headlight ambient=".5 .5 .5" diffuse=".7 .7 .7"/></visual><asset>
      <mesh name="person" file={mesh}/>
      <texture name="skin" type="2d" file={texture}/>
      <material name="skin" texture="skin" specular=".02"/></asset><worldbody>
      <light pos="0 -3 3" dir="0 1 -.2" diffuse=".7 .7 .7"/>
      <geom type="plane" size="10 10 .1" rgba=".6 .6 .55 1"/>
      <body name="person"><geom type="mesh" mesh="person" material="skin" contype="0" conaffinity="0"/></body>
      <geom name="background" type="box" pos="0 1 1" size="4 .1 1.3" rgba=".7 .63 .5 1"/>
      <camera name="registration" pos="0 {-distance} {height*.5}" xyaxes="1 0 0 0 0 1" fovy="45"/>
      </worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    person=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'person')
    wall=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'background')
    with mujoco.Renderer(model,height=640,width=640) as renderer:
        for angle in angles:
            model.body_quat[person]=[math.cos(math.radians(angle)/2),0,0,math.sin(math.radians(angle)/2)]
            if background:model.geom_rgba[wall,:3]=[.3,.45,.6]
            mujoco.mj_forward(model,data);renderer.update_scene(data,'registration')
            yield angle,cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)


def enroll(identity,height,detector):
    embeddings=[];colours=[];views=[]
    for angle,image in render_views(identity,height,angles=tuple(range(0,360,30))):
        rows,_=detector.identify(image,{})
        if len(rows)!=1:continue
        embeddings.append(rows[0]['embedding']);colours.append(rows[0]['colour']);views.append(angle)
    if len(embeddings)<4:raise ValueError(f'{identity}: insufficient detected registration views: {views}')
    return {'embeddings':embeddings,'colours':colours,'views_deg':views,'model':'OSNet x0.25 MSMT17 + low-weight HSV bands',
            'source':'independent multi-view asset registration; no online actor-state input'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity',required=True);p.add_argument('--height',type=float,default=1.64)
    a=p.parse_args()
    from cli import identity
    name=identity(a.identity)
    if not math.isfinite(a.height) or not .8<=a.height<=2.3:raise ValueError('Invalid registration height')
    if name not in read_json(DATA/'face-gallery.json',{}):raise ValueError('Enroll the face and voice first')
    for filename in ['person-raised.obj','person.png']:
        if not (ROOT/'assets/avatars'/name/filename).is_file():raise ValueError('Missing avatar: '+filename)
    rows=[dict(id=name,height=a.height)]
    gallery=read_json(DATA/'body-gallery.json',{});detector=Bodies()
    for row in rows:
        gallery[row['id']]=enroll(row['id'],row.get('height',1.64),detector)
        print(row['id'],len(gallery[row['id']]['embeddings']),'views',flush=True)
        write_json(DATA/'body-gallery.json',gallery)


if __name__=='__main__':main()
