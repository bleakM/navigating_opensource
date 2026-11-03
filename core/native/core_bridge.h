#pragma once
// SIMULATOR SIDE ONLY. Actor staging, no robot pose commands.
#include <mujoco/mujoco.h>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>

inline void WorkbenchTick(const mjModel* model,mjData* data) {
  const char* directory=std::getenv("XIAOBEI_WORKBENCH");
  if (!directory) return;
  static auto last=std::chrono::steady_clock::now();
  static double face_at[4]={-100.,-100.,-100.,-100.};
  static bool was_visible[4]={false,false,false,false};
  auto now=std::chrono::steady_clock::now();
  if (std::chrono::duration<double>(now-last).count()<.08) return;
  last=now;
  std::string base(directory);
  try {
    auto command=YAML::LoadFile(base+"/actors.json");
    int locked=command["locked_slot"] ? command["locked_slot"].as<int>() : -1;
    for (const auto& actor:command["actors"]) {
      int slot=actor["slot"].as<int>();
      if(slot<0 || slot>=4) continue;
      int id=mj_name2id(model,mjOBJ_BODY,("wb_actor_"+std::to_string(slot)).c_str());
      if(id<0 || model->body_mocapid[id]<0) continue;
      int mocap=model->body_mocapid[id];
      if(slot!=locked) {
        double x=actor["x"].as<double>(),y=actor["y"].as<double>();
        if(std::isfinite(x)&&std::isfinite(y)&&std::abs(x)<1000&&std::abs(y)<1000) {
          data->mocap_pos[3*mocap]=x;data->mocap_pos[3*mocap+1]=y;
        }
      }
    }
  } catch(...) { /* No command file at startup is normal. */ }
  // Face the camera on entering its unobstructed view, then refresh every 3 s
  // while remaining visible. A global 3 s poll used to miss a whole fast sweep.
  // This authors the world, not recognition or caller identity.
  int torso=mj_name2id(model,mjOBJ_BODY,"torso_link");
  if(torso>=0) {
    const mjtNum* p=data->xpos+3*torso;const mjtNum* r=data->xmat+9*torso;
    mjtByte group[mjNGROUP]={0,0,1,0,0,0};
    for(int slot=0;slot<4;slot++) {
      int id=mj_name2id(model,mjOBJ_BODY,("wb_actor_"+std::to_string(slot)).c_str());
      if(id<0 || model->body_mocapid[id]<0) continue;
      int m=model->body_mocapid[id];
      double dx=data->mocap_pos[3*m]-p[0],dy=data->mocap_pos[3*m+1]-p[1];
      double forward=r[0]*dx+r[3]*dy,left=r[1]*dx+r[4]*dy;
      // 640x480 and vertical FOV45: horizontal half-angle about .505 rad.
      if(forward<=0 || std::abs(std::atan2(left,forward))>.505) {was_visible[slot]=false;continue;}
      double origin[3]={p[0]+.13*r[0],p[1]+.13*r[3],p[2]+.40};
      double direction[3]={dx,dy,0};mju_normalize3(direction);int hit=-1;
      double distance=mj_ray(model,data,origin,direction,group,1,-1,&hit);
      if(distance>=0 && distance<std::hypot(dx,dy)-.25) {was_visible[slot]=false;continue;}
      bool entered=!was_visible[slot];was_visible[slot]=true;
      if(!entered && data->time-face_at[slot]<3.) continue;
      face_at[slot]=data->time;
      double angle=std::atan2(-dy,-dx)+mjPI/2;
      data->mocap_quat[4*m]=std::cos(angle/2);data->mocap_quat[4*m+1]=0;
      data->mocap_quat[4*m+2]=0;data->mocap_quat[4*m+3]=std::sin(angle/2);
    }
  }
}
