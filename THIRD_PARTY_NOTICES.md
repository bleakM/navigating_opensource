# Third-party notices

Independent integration scripts and configuration: Apache-2.0 (`LICENSE`).

| Dependency | Source | Terms |
|---|---|---|
| Grove-G1 / Canopy | https://github.com/Adyansh04/grove-g1 ; https://github.com/Adyansh04/canopy | BSD-3-Clause; pinned sources and patches in `g1/sources.lock.json`, `g1/patches`; see `licenses/LICENSE-GROVE` |
| Unitree MuJoCo / SDK2 | https://github.com/unitreerobotics/unitree_mujoco ; https://github.com/unitreerobotics/unitree_sdk2 | BSD-3-Clause; original notices remain in fetched source |
| FAST-LIO humanoid fork | https://github.com/Adyansh04/FAST_LIO_LOCALIZATION_HUMANOID | GPL-2.0; fetched separately, not relicensed by this repository |
| Pretrained AGILE controller | Grove `g1_controllers/policy` | Separate `LICENSE.WBC-AGILE`; no weights redistributed |
| RTAB-Map / Nav2 / ROS 2 | https://github.com/introlab/rtabmap_ros ; https://github.com/ros-navigation/navigation2 | Upstream package licenses |
| CAMPPlus / sherpa-onnx | https://github.com/modelscope/3D-Speaker ; https://github.com/k2-fsa/sherpa-onnx | Apache-2.0; model manifest records hashes |
| YuNet / SFace | https://github.com/opencv/opencv_zoo | MIT / Apache-2.0; license texts in `licenses/` |
| Silero VAD | https://github.com/snakers4/silero-vad | MIT; `licenses/LICENSE-SILERO` |
| pyroomacoustics | https://github.com/LCAV/pyroomacoustics | MIT |
| Microsoft Rocketbox | https://github.com/microsoft/Microsoft-Rocketbox | MIT; avatars not included |
| Furnished scene assets | Grove `workspace/assets/manifest.yaml` | Per-asset attribution (including Poly Haven CC0 and CC BY assets); meshes/textures not included |

`examples/exhibition` is a scene description derived from the Grove scene generator; required external furniture assets retain their upstream licenses. `core/build_native.py` modifies separately fetched BSD simulator sources in an ignored build directory: actor/sensor separation and native rendering are the integration changes. Source headers and upstream notices must be retained when distributing resulting binaries.

No YOLO model or detector is shipped: the final identity pipeline uses face-only YuNet/SFace. Older body/gesture detection experiments are outside this source release.
