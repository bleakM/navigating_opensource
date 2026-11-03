# Source release validation · 2026-11-03

Validation uses the existing prepared Ubuntu 24.04 / ROS 2 Jazzy dependency root on an Ubuntu 22.04 host. No host dependencies were upgraded.

| Check | Actual outcome |
|---|---|
| Python syntax and shell syntax | Passed |
| Native MuJoCo actor/sensor adapter | Compiled successfully from pinned upstream sources |
| Core safety/scheduling/acoustic regressions | **51 tests passed** |
| Recognition-model hashes | Four identity/VAD ONNX files and four Chinese KWS files verified |
| Actual pretrained model initialization | CAMPPlus, YuNet, SFace and Chinese wake word loaded |
| CLI face + voice enrollment | Passed with one detected rendered-avatar face and a 2.544 s existing call recording; speech duration 2.402 s, recording RMS −26.5 dBFS |
| Saved-map session, without any web service | Real RTAB-Map registration passed; Nav2 and navigation-map lifecycle active; fresh map and estimated map-frame pose available |
| Actual short navigation | Nav2 succeeded at goal `[0.8, 0.0]`; estimated final XY `[0.59582, −0.11204]`; error **0.23290 m**, FAST-LIO XY integral **0.68473 m**; 34 path poses, path length **0.93336 m** |
| Upper-body control | Existing arm trajectory controller activated; measured during this short session, maximum requested shoulder swing 0.2645 rad; leg policy unchanged |

The navigation metrics come from the unmodified core goal checker, using estimated TF/odometry. A short local navigation test is not a cross-room endurance result. Waveform regressions cover three controlled free-field bearings, level adjustment and silence rejection; these are not real-building acoustic tests.

During extraction, an empty optional `initial_pose` was initially passed as a malformed ROS launch argument. The CLI now omits the argument when empty; the subsequent actual session and navigation succeeded. The added acoustic test initially used continuous noise, which the speech-frame gate correctly rejected; the fixture now has quiet intervals. Algorithm thresholds were not relaxed.

The private source revision preceding extraction was `10e8d16e2ec6a6a30dacabd18d9c4e871a741f46`. Its final exhibition experiment had already completed fresh mapping and a specified three-person saved-map search sequence. Those historical runs are not added to the counts above. The source-only extraction has **not** rerun a full empty-map exploration or a full three-person end-to-end sequence. Fresh-machine installation has also not been fully reproduced.

Runtime exploration no longer depends on offline simulator-ground-truth scoring scripts. It preserves the upstream exploration tree, map saving and localization gate; it explicitly reports coverage as independently unmeasured. This keeps evaluation data outside operational planning.

No private voice waveform, face image, biometric gallery, map database, model weight, complete rosbag or old report is published with these checks.
