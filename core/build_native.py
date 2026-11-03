#!/usr/bin/env python3
"""Build a separate adapter binary; never modify/install over the frozen simulator."""
import shutil,subprocess,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
dependency=Path(os.environ.get('XIAOBEI_DEPENDENCIES') or (json.loads((ROOT/'dependencies.local.json').read_text())['root'] if (ROOT/'dependencies.local.json').is_file() else ROOT))
src=dependency/'third_party/grove-g1/workspace/vendor/unitree_mujoco/simulate'
out=ROOT/'.runtime/core-native'
out.mkdir(exist_ok=True)
shutil.copytree(src/'src',out/'src',dirs_exist_ok=True)
shutil.copy2(src/'CMakeLists.txt',out/'CMakeLists.txt')
shutil.copy2(src/'config.yaml',out/'config.yaml')
if not (out/'mujoco').is_symlink() and not (out/'mujoco').exists():(out/'mujoco').symlink_to('/opt/unitree_robotics/mujoco-3.3.6')
shutil.copy2(ROOT/'core/native/core_bridge.h',out/'src/core_bridge.h')
p=out/'src/main.cc';s=p.read_text();s=s.replace('#include "sensor_publisher.h"','#include "sensor_publisher.h"\n#include "core_bridge.h"')
s=s.replace('if (m)\n        {','if (m)\n        {\n          WorkbenchTick(m,d);',1);p.write_text(s)

p=out/'src/sensor_publisher.cc';s=p.read_text();s=s.replace('cam_cam.fixedcamid = rig.cam_id;','cam_opt.geomgroup[4] = (rig.name == "interaction");\n                cam_cam.fixedcamid = rig.cam_id;')
# Software GL in this runtime: extra shadow passes make two sensor streams stale.
# Geometry, textures, camera calibration, depth and the physics engine are retained.
s=s.replace('mjr_render(vp, &cam_scn, &cam_con);','cam_scn.flags[mjRND_SHADOW] = (rig.name != "interaction");\n                cam_scn.flags[mjRND_REFLECTION] = (rig.name != "interaction");\n                mjr_render(vp, &cam_scn, &cam_con);');p.write_text(s)
# Keep the upstream raycasting and camera implementations, but schedule them independently.
# Otherwise a slow offscreen camera also holds back every LiDAR packet and FAST-LIO TF.
def replace_once(old,new):
    global s
    if s.count(old)!=1:raise RuntimeError('Upstream sensor adapter anchor changed: '+old[:80])
    s=s.replace(old,new,1)
replace_once('std::thread                  imu_thread;', 'std::thread                  camera_thread;\n    std::thread                  imu_thread;')
replace_once('RelaySocket* relay_socket)\n{\n    // The model is loaded', 'RelaySocket* relay_socket, bool camera_only)\n{\n    // The model is loaded')
replace_once('for (const CameraConfig& camera : cfg.cameras) {', 'for (const CameraConfig& camera : cfg.cameras) {\n        if (!camera_only) break;')
replace_once('// Sent before the sweep, so they are ~32 ms fresher.', 'if (!camera_only) {\n        // Sent before the sweep, so they are ~32 ms fresher.')
replace_once('if (cam_win != nullptr && !rigs.empty() && !reload_pending) {', 'relay.send(header, points.data(), points.size() * sizeof(float));\n        }\n\n        if (cam_win != nullptr && !rigs.empty() && !reload_pending) {')
replace_once('\n        relay.send(header, points.data(), points.size() * sizeof(float));\n\n        std::this_thread::sleep_until(next);', '\n        std::this_thread::sleep_until(next);')
replace_once('state().thread     = std::thread(sensorLoop, cfg, model, data, sim_mtx, state().relay.get());', 'state().thread = std::thread(sensorLoop, cfg, model, data, sim_mtx, state().relay.get(), false);\n    state().camera_thread = std::thread(sensorLoop, cfg, model, data, sim_mtx, state().relay.get(), true);')
replace_once('if (s.imu_thread.joinable()) {', 'if (s.camera_thread.joinable()) s.camera_thread.join();\n    if (s.imu_thread.joinable()) {')
p.write_text(s)
subprocess.run([str(ROOT/'scripts/g1-runtime'),'cmake','-S','/workspace/.runtime/core-native','-B','/workspace/.runtime/core-native/build','-DCMAKE_BUILD_TYPE=Release'],check=True)
subprocess.run([str(ROOT/'scripts/g1-runtime'),'cmake','--build','/workspace/.runtime/core-native/build','--target','unitree_mujoco','-j2'],check=True)
