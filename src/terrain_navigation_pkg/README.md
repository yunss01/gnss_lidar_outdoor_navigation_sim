# Terrain navigation

This package contains the current outdoor-navigation stack. It uses GNSS
waypoints, odometry/IMU state, and 3D LiDAR obstacle geometry to drive a
forward-only Ackermann vehicle. Camera/IMU roughness prediction is not part of
this repository's navigation architecture.

The planner policy is defined in
[`PLANNING_DESIGN.md`](PLANNING_DESIGN.md) documents the safety boundary,
rolling local map, Nav2 planning, and GNSS waypoint policy.

## Navigation learning recorder

`terrain_navigation_nav2.launch.py` starts a subscriber-only recorder by
default. Pressing F9 or F10 starts a session automatically; completion,
failure, cancellation, or shutdown closes it. The recorder never publishes a
control command, and compressed file I/O runs on a background thread.

```text
~/terrain_nav_data/learning/raw/session_YYYYMMDD_HHMMSS_microseconds/
  metadata.json
  frames.csv
  samples/sample_000001.npz
```

Each sample contains the clipped raw 3D points, four-channel vehicle-centric
LiDAR BEV, local/global costmap crops, Nav2 plan, long-range guide and subgoal,
goal direction, odometry, teacher commands, and safety state. This preserves
both successful and failed teacher behavior for later filtering and local
policy training. Disable recording for a run with:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  record_learning_data:=false
```

Before training, build a non-destructive session-grouped manifest. The
command audits the raw recordings, verifies five evenly spaced NPZ samples
per session, separates clean imitation frames from recovery/negative frames,
and reports route leakage between the development splits:

```bash
ros2 run terrain_navigation_pkg build_learning_manifest
```

Derived files are written under
`~/terrain_nav_data/learning/manifests/v1/`. The raw session directories are
never modified. `manifest_train.csv`, `manifest_validation.csv`, and
`manifest_test.csv` contain only conservative clean-imitation candidates;
`manifest_recovery.csv` and `manifest_safety_negative.csv` are retained
separately for later DAgger, safety, and recovery-policy work. The default
session split is only for initial pipeline development. A final generalization
experiment must collect enough distinct routes and rebuild with
`--split-unit route`.

Create fixed-spacing trajectory targets and diagnostic contact sheets with:

```bash
ros2 run terrain_navigation_pkg visualize_learning_trajectories
```

The default command analyzes train and validation only; it deliberately does
not expose the test split during model development. It resamples each Nav2
teacher path at 0.75 m spacing into 12 masked targets, reports suspicious
curvature, winding, short-horizon, direction, and temporal-change cases, and
writes review images to
`~/terrain_nav_data/learning/visualizations/v1/`. Diagnostic flags never
delete or relabel raw recordings automatically.

After visual review, the first behavior-cloning dataset excludes only paths
flagged as `winding_path`. Full target archives remain available for audit and
recovery experiments, while the filtered files are named
`trajectory_targets_clean_train.npz` and
`trajectory_targets_clean_validation.npz`. Load them with PyTorch as follows:

```python
from torch.utils.data import DataLoader
from terrain_navigation_pkg.navigation_learning_dataset import (
    NavigationTrajectoryDataset,
)

dataset = NavigationTrajectoryDataset(
    '~/terrain_nav_data/learning/visualizations/v1/'
    'trajectory_targets_clean_train.npz'
)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
batch = next(iter(loader))
```

Each item contains a normalized four-channel `lidar_bev`, raw and normalized
relative goals, raw and normalized 12-point trajectories, a Boolean
`target_mask`, and vehicle speed. The loader rejects an archive containing an
initial behavior-cloning exclusion flag unless explicitly overridden.

The CNN baseline predicts the 12 XY targets from the four-channel BEV and the
normalized relative goal. Its current spatial encoder adds forward/left
coordinate channels and retains a coarse 5-by-5 feature grid instead of
globally averaging away obstacle position. Run a short pipeline smoke test
with:

```bash
ros2 run terrain_navigation_pkg train_navigation_trajectory_model \
  --epochs 1 --batch-size 8 --num-workers 0 \
  --max-train-batches 2 --max-validation-batches 1 --device cpu \
  --output-directory ~/terrain_nav_data/learning/models/trajectory_baseline/smoke
```

For an actual training run, remove the `max-*` options and use
`--device cuda` after confirming that the PyTorch installation can access the
RTX 3060. The model is an offline baseline at this stage; it is not connected
to the running vehicle or Nav2 control loop.

Evaluate the trained checkpoint on the held-out validation split with:

```bash
ros2 run terrain_navigation_pkg evaluate_navigation_trajectory_model \
  --checkpoint ~/terrain_nav_data/learning/models/trajectory_baseline/v3/best.pt \
  --device cuda \
  --output ~/terrain_nav_data/learning/models/trajectory_baseline/v3/validation_metrics.json \
  --predictions-output ~/terrain_nav_data/learning/models/trajectory_baseline/v3/validation_predictions.npz
```

The reported coordinate, point-distance, and endpoint errors are geometric
offline metrics. They do not yet imply that the learned path is safe to send
to the vehicle controller.

Overlay the exported predictions and teacher trajectories on the recorded
LiDAR BEV with:

```bash
ros2 run terrain_navigation_pkg visualize_navigation_trajectory_predictions
```

The command writes random and worst-error contact sheets under
`~/terrain_nav_data/learning/models/trajectory_baseline/v3/`
`prediction_visualizations/`.

## Static-obstacle traversability learning

The next learning stage treats road, paving, grass, soil, and mildly uneven
surfaces as the same free-space class.  It does not optimize ride comfort.
The target map contains only free, static-obstacle, and unknown cells.  During
CARLA data collection, a second semantic LiDAR supplies privileged labels;
the deployed model still receives only the normal geometric 3D LiDAR.

Start the CARLA client with training labels enabled only while collecting this
dataset:

```bash
python3 ~/carla/0.10.0/PythonAPI/examples/manual_control.py \
  --external-control --semantic-lidar-labels
```

Then start navigation recording with raw tensors and semantic labels:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true record_learning_data:=true \
  collect_traversability_data:=true
```

The semantic scan is accepted only when its completed-revolution timestamp is
within 30 ms of the normal LiDAR scan.  Every accepted sample stores semantic
XYZ, incidence cosine, instance ID, semantic tag, and current IMU state beside
the geometric BEV.  Semantic tags are training supervision and must never be
added to the model input.  Normal F9/F10 runs keep this option disabled because
the second ray-cast sensor approximately doubles LiDAR simulation work.

After collecting one or more routes, create the model-ready input/target pairs:

```bash
ros2 run terrain_navigation_pkg build_traversability_dataset
```

The command reads raw sessions without modifying them and writes derived files
to `~/terrain_nav_data/learning/traversability/v1/`.  Each derived NPZ contains
the four-channel geometric `lidar_bev`, a `target_labels` grid with unknown
(`-1`), free (`0`), and obstacle (`1`) cells, supporting-count grids, and the
recorded pose/goal state.  CARLA object tags are deliberately omitted from the
derived model inputs.  `manifest.csv` records accepted and skipped samples, and
`summary.json` records the class policy and target-generation parameters.

The nominal Nav2 body footprint (`x=-2.5..2.4 m`, `|y|<=1.0 m`) is always
masked to unknown so roof/hood returns from the ego vehicle cannot become
obstacle targets.  Ray-cleared cells are stored separately as
`visibility_free_mask` and `visibility_ray_count`; they are not merged into
the semantic target.  A 3-D ray may pass above a low obstacle, so treating all
projected ray cells as semantic free space would create unsafe supervision.
The separate visibility evidence is intended for conservative accumulated-map
fusion after the endpoint classifier has been validated.

The evidence-v2 target contract, controlled-actor capture procedure,
within-session exact-scan deduplication, actor-absent counterfactual audit, and
current motorhelmet pilot results are documented in
[`TRAVERSABILITY_EVIDENCE_V2.md`](TRAVERSABILITY_EVIDENCE_V2.md). V2 is an
audited research dataset with a current-only 8-channel online shadow node.
The v2 node publishes only isolated `/learning/evidence_v2/*` topics; it has
not replaced the online v1 model, the Nav2 obstacle cloud, or any safety or
vehicle-command input and has not been granted control authority.

Audit the completed-session targets before collecting more data or training:

```bash
ros2 run terrain_navigation_pkg visualize_traversability_dataset
```

The Pillow-based tool avoids the system Matplotlib/NumPy ABI dependency and
writes deterministic random and route-sequence contact sheets as 300-DPI PNG
and PDF files under
`~/terrain_nav_data/learning/traversability/visualizations/v1/`.  Each sample
shows geometric LiDAR, direct free/obstacle labels, separate ray-visibility
evidence, and an audit overlay that highlights height-span obstacle overrides.

Train the first feasibility model with a whole recording session reserved for
validation:

```bash
ros2 run terrain_navigation_pkg train_traversability_model \
  --validation-session session_20260915_131243_896751
```

The model is a compact U-Net endpoint classifier. Its only external input is
the normalized four-channel geometric LiDAR BEV; semantic tags and GNSS are
not model inputs. Unknown target cells are excluded from the loss. The
training command writes `best.pt`, `last.pt`, `history.csv`, and an auditable
`split_manifest.csv` under
`~/terrain_nav_data/learning/models/traversability_pilot/v1/`. Splitting is
performed by complete session, never by randomly mixing neighboring frames.
This Town10 result is a pipeline feasibility check rather than evidence of
generalization to grass, mine terrain, or a real vehicle.

Evaluate the reserved session and optionally sample dropout uncertainty:

```bash
ros2 run terrain_navigation_pkg evaluate_traversability_model \
  --checkpoint ~/terrain_nav_data/learning/models/traversability_pilot/v1/best.pt \
  --manifest ~/terrain_nav_data/learning/traversability/v1/manifest.csv \
  --session session_20260915_131243_896751 \
  --output ~/terrain_nav_data/learning/models/traversability_pilot/v1/validation_mc8.json \
  --mc-samples 8
```

The report includes free/obstacle IoU, obstacle precision and recall,
calibration error, predictive entropy, and Monte-Carlo dropout variance.
An untouched session or map must be collected later for a final test; the
validation session must not be repeatedly tuned and then reported as a test.

The domain-augmented pilot is trained separately so the clean v1 checkpoint
is never overwritten:

```bash
ros2 run terrain_navigation_pkg train_traversability_model \
  --validation-session session_20260915_131243_896751 \
  --sensor-augmentation-probability 0.8 \
  --maximum-height-bias-m 0.20 --maximum-tilt-deg 4.0 \
  --height-noise-std-m 0.03 --minimum-density-scale 0.50 \
  --maximum-cell-dropout 0.20 \
  --output-directory ~/terrain_nav_data/learning/models/traversability_pilot/v2_domain_aug
```

Synthetic stress tests expose sensitivity to mounting-height error,
pitch/roll, density change, height noise, and missing cells:

```bash
ros2 run terrain_navigation_pkg stress_test_traversability_model \
  --checkpoint ~/terrain_nav_data/learning/models/traversability_pilot/v2_domain_aug/best.pt \
  --manifest ~/terrain_nav_data/learning/traversability/v1/manifest.csv \
  --output ~/terrain_nav_data/learning/models/traversability_pilot/v2_domain_aug/stress_test.json
```

These perturbations are diagnostic tests, not a replacement for a new map or
real-vehicle test. Since they were selected after inspecting validation
behavior, their results must not be reported as an independent test result.

Inspect random predictions and the frames with the highest missed-obstacle
rate before connecting the model to navigation:

```bash
ros2 run terrain_navigation_pkg visualize_traversability_predictions \
  --checkpoint ~/terrain_nav_data/learning/models/traversability_pilot/v1/best.pt \
  --manifest ~/terrain_nav_data/learning/traversability/v1/manifest.csv \
  --output-directory ~/terrain_nav_data/learning/models/traversability_pilot/v1/prediction_visualizations
```

Magenta cells are semantic obstacle targets predicted as free and are the
safety-critical error. Blue cells are free targets predicted as obstacles.
Predictions are shown only for directly supervised cells; empty/unknown space
must remain unknown until separate ray-clearing evidence is fused.

Run the checkpoint online in **shadow mode** before granting it any map or
control authority:

```bash
ros2 run terrain_navigation_pkg traversability_shadow_node --ros-args \
  -p checkpoint_path:=/home/sukja/terrain_nav_data/learning/models/traversability_pilot/v2_domain_aug/best.pt \
  -p device:=auto \
  -p mc_samples:=4
```

The node subscribes to `/lidar/points` and publishes only isolated diagnostic
topics:

- `/learning/traversability_probability`: obstacle probability on observed
  cells
- `/learning/traversability_uncertainty`: normalized predictive entropy
- `/learning/traversability_shadow_decision`: selective free/obstacle/unknown
  result
- `/learning/traversability_shadow_status`: JSON timing and cell counts

None of these topics is consumed by Nav2, the safety gate, or vehicle command
nodes. A high-confidence learned-free decision cannot erase a return above
the fixed hard-obstacle height or a cell with at least 0.15 m vertical span.
Low-confidence and high-variance returns remain unknown rather than free.
This rule is a deployment guard, not proof of out-of-distribution detection;
real-vehicle scans must first be observed and logged in shadow mode.

When shadow inference is started from the integrated launch, a passive fusion
node also publishes `/learning/nav2_obstacles_candidate`. This PointCloud2 is
the exact interface candidate intended for a later Nav2-only pilot, but it has
no navigation or safety authority. It combines three sources conservatively:

- a confident learned-free cell may remove a point from the existing
  `/lidar/nav2_obstacles` baseline;
- a learned-obstacle cell may add a low-height raw LiDAR endpoint;
- geometric hard-obstacle cells veto learned clearing but remain under the
  established baseline filter instead of bulk-adding tall raw returns;
- an uncertain, unobserved, or out-of-BEV cell retains the existing baseline
  behavior.

Raw LiDAR, baseline obstacle cloud, and learned decision must carry the same
sensor timestamp before a candidate is emitted. A missing or unmatched AI
frame produces no candidate; the still-independent baseline remains active,
and a later authority mux must explicitly select that baseline. The companion
`/learning/obstacle_candidate_status` topic reports source counts and cache
diagnostics. The candidate topic is deliberately not listed as an observation
source in either Nav2 costmap. In RViz it appears in green as `AI fused
obstacle candidate (NO AUTHORITY)` for side-by-side inspection with the red
baseline obstacle cloud. Disable it explicitly with
`traversability_candidate_enabled:=false` if only the grids are required.

Before granting the candidate any Nav2 authority, run the fixed-duration
static-obstacle audit described in `STATIC_OBSTACLE_AUDIT.md`. The audit node
compares the baseline and candidate clouds against synchronized CARLA semantic
targets while the vehicle remains stationary and `drive_enabled` is false:

```bash
ros2 run terrain_navigation_pkg \
  traversability_static_obstacle_audit_node --ros-args \
  -p scenario_label:=wall_center_5m \
  -p obstacle_type:=wall \
  -p obstacle_distance_m:=5.0 \
  -p obstacle_lateral_position:=center
```

It writes `metadata.json`, per-frame CSV metrics, `summary.json`, and only a
few compact diagnostic NPZ snapshots below
`~/terrain_nav_data/learning/static_obstacle_audits/`. The primary unsafe
change is `removed_target_obstacle`: a semantically verified obstacle cell
that existed in the baseline cloud but disappeared from the candidate.

For route-level shadow validation, start the integrated Nav2 launch.  The
recorder is enabled automatically whenever shadow inference is enabled:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true \
  guide_mode:=far \
  record_learning_data:=false \
  traversability_shadow_enabled:=true
```

Each F9/F10 route creates one directory below
`~/terrain_nav_data/learning/shadow_runs/`.  `frames.csv` stores confidence,
coverage, timing, waypoint, speed, and navigation/safety state for every
shadow frame.  `summary.json` contains route aggregates.  Compressed grid
snapshots are saved every 10 seconds, on waypoint changes, and during
high-uncertainty events, with a maximum of 64 snapshots per route.  These
diagnostics do not change the Nav2 costmaps, safety decisions, or commands.

In CARLA, privileged semantic LiDAR can be used to measure actual shadow-map
accuracy over a route. Start `manual_control.py` with
`--semantic-lidar-labels`, then enable the isolated evaluator:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true \
  guide_mode:=far \
  record_learning_data:=false \
  traversability_shadow_enabled:=true \
  traversability_shadow_evaluation_enabled:=true
```

The evaluator time-aligns the probability grid, selective-decision grid, and
semantic scan within 30 ms. It uses the same semantic target builder as the
offline dataset and evaluates only directly supervised cells. Results are
written under `~/terrain_nav_data/learning/shadow_evaluations/`. Raw model
IoU, precision, recall, and false-free/false-obstacle rates are reported
separately from selective coverage, accepted accuracy, abstention, and the
safety-critical selective false-free rate. Counts represent repeated cell
observations over time, not statistically independent world-map cells.

The evaluator also subscribes to `/safety/checked_trajectory`, whose header
matches the LiDAR scan used by the independent safety gate. Oriented vehicle
rectangles are rasterized along this Ackermann path to form a path-critical
swept corridor. Corridor coverage, false-free/false-obstacle rates, maximum
false-free cluster size and consecutive affected frames are reported
separately from the full-BEV values. If a matching checked trajectory does not
arrive within 120 ms, a straight 4 m fallback is used and identified in the
CSV and summary; it is never silently presented as the commanded path.

Every frame containing at least one selective false-free cell saves a compact
NPZ below the route's `error_snapshots/` directory. It contains the semantic
target, raw probability, selective decision, swept-corridor mask, both raw and
selective false-free masks, target support counts, vertical span and sampled
trajectory. These files are evaluation artifacts only and are not consumed by
the model or navigation stack.

`/lidar/semantic_points` is read only by this evaluator. It is not passed to
the network, Nav2, the safety gate, or a vehicle command topic. Keep online
semantic evaluation disabled outside CARLA or when the extra semantic sensor
is unavailable. Because the additional ray-cast sensor and evaluator add
simulation load, use these runs for perception accuracy only. Do not mix their
completion time or stopped time with the normal F9/F10 navigation benchmark.

## GNSS goal manager

`gnss_goal_manager_node` subscribes to:

- `/gnss/fix` (`sensor_msgs/NavSatFix`, AKN-940-LTE + AKA900)
- `/vehicle/odometry` (`nav_msgs/Odometry`)
- `/navigation/goal_gnss` (`sensor_msgs/NavSatFix`)

It fixes the first valid GNSS sample as a local WGS84 ENU origin and publishes:

- `/navigation/current_local`
- `/navigation/goal_local`
- `/navigation/goal_pose`
- `/navigation/goal_vector`
- `/navigation/distance_to_goal`
- `/navigation/bearing_to_goal_deg`
- `/navigation/goal_reached`
- `/navigation/status`
- `/navigation/direct_path`

`direct_path` is only the unobstructed straight-line goal reference. It must
not be treated as a collision-free global plan. A map-based planner will
replace it in the next stage.

Set a goal after the node starts:

```bash
ros2 topic pub --once /navigation/goal_gnss sensor_msgs/msg/NavSatFix \
  "{latitude: 37.0, longitude: 127.0, altitude: 50.0}"
```

Or start the goal manager and low-speed GPS controller together:

```bash
ros2 launch launch_pkg terrain_navigation.launch.py \
  goal_enabled:=true \
  goal_latitude:=37.0 \
  goal_longitude:=127.0 \
  goal_altitude:=50.0
```

The legacy launch keeps the optional 3D terrain mapper disabled. It is not
required for the first A-to-B test.

## Phase 1: low-speed A-to-B test

### CARLA real-hardware sensor profile

`manual_control.py` defaults to a sensor-interface profile that keeps the
Lincoln body/dynamics but presents the planned real hardware contracts:

- OS0-32U-like 32-channel, 10 Hz LiDAR using this unit's metadata elevation
  endpoints (`-45.85` to `+42.97` degrees) on
  `/lidar/points` with frame `os_lidar`;
- VN-200 IMU-only output on `/vectornav/imu` with frame `vn200_link`;
- AKN-940-LTE plus AKA900 GNSS on `/gnss/fix` with frame `gnss_antenna`.

The default `realtime` OS0 density preserves the previously stable CARLA ray
budget of 160,000 rays/s.  `--os0-density exact` requests
`32 * 1024 * 10 = 327,680` rays/s, but it may reduce pygame/server real-time
performance.  CARLA's uniformly spaced ray caster is only OS0-like; it does
not reproduce the unit-specific beam calibration, returns, intensity, timing,
or organized `32 x 1024` packet layout of the physical sensor.

GNSS defaults to `--gnss-quality standalone`.  Use `rtk_fixed` only after the
real correction service has been demonstrated; otherwise the simulation would
hide a deployment dependency.  The AKN-940-LTE datasheet does not specify
vertical accuracy, so the simulated vertical covariance is deliberately
conservative and provisional.

The Lincoln-relative sensor mounts are also provisional.  Replace them only
after measuring `base_link` to `os_lidar`, `vn200_link`, and the AKA900 antenna
phase center on the physical vehicle.  The AKN receiver-box position is not a
GNSS measurement origin.

The matching topic names and frames validate the software interfaces, not a
complete real-vehicle localization stack.  In the current CARLA profile,
`/vehicle/odometry` is simulator ground truth and is the position/heading used
by Nav2 and the GNSS goal manager after its initial GNSS anchor.  `/vectornav/imu`
is recorded for learning/diagnostics but is not yet fused into odometry.  Real
deployment therefore still requires measured extrinsics, time synchronization,
and a GNSS/IMU/vehicle-odometry estimator before simulator ground truth can be
removed.

Recommended first regression run:

```bash
python3 ~/carla/0.10.0/PythonAPI/examples/manual_control.py \
  --filter 'vehicle.lincoln.mkz*' \
  --sync \
  --fixed-delta-seconds 0.05 \
  --external-control \
  --sensor-profile real_vehicle \
  --os0-density realtime \
  --gnss-quality standalone
```

This navigation regression deliberately keeps the proven `0.05 s` physics
step.  CARLA emits at most one sample per simulation tick, so the nominal
VN-200 `0.01 s` sensor tick is effectively limited to **20 Hz**, not 100 Hz,
in this mode.  Use `--fixed-delta-seconds 0.01` for a separate 100 Hz
timestamp/rate stress test; it may run slower than wall time and is not the
first F9 acceptance test.  The simulated IMU adds a fixed per-run attitude
bias and gyro bias, but it still does not reproduce temperature drift,
magnetic interference, vibration, or the full VN-200 estimator dynamics.

Use `--sensor-profile legacy` only as an A/B reference against the previous
LiDAR FoV and optimistic GNSS-aided VN-200 covariance.  Topic and frame names
stay identical so the comparison does not silently disconnect a subscriber.
A legacy route completion is a baseline, not evidence that the real-hardware
profile is correct.

First load a normal CARLA town instead of Mine. This CARLA installation ships
the rendered `Town10HD_Opt` map; its streets, buildings, intersections, and
open areas are suitable for the later waypoint loop:

```bash
python3 ~/carla/0.10.0/PythonAPI/util/config.py --map Town10HD_Opt
```

Start the vehicle client in explicit ROS control mode. This mode cannot be
combined with CARLA autopilot, and a missing `/cmd_vel` applies full brake
after 0.5 seconds:

```bash
source /opt/ros/humble/setup.bash
python3 ~/carla/0.10.0/PythonAPI/examples/manual_control.py \
  --filter 'vehicle.lincoln.mkz*' \
  --sync \
  --fixed-delta-seconds 0.05 \
  --external-control \
  --sensor-profile real_vehicle \
  --os0-density realtime \
  --gnss-quality standalone
```

Then launch navigation with the desired WGS84 coordinate:

```bash
source /opt/ros/humble/setup.bash
source ~/terrain_nav_ws/install/setup.bash
ros2 launch launch_pkg terrain_navigation.launch.py \
  goal_enabled:=true \
  goal_latitude:=DESTINATION_LATITUDE \
  goal_longitude:=DESTINATION_LONGITUDE \
  goal_altitude:=DESTINATION_ALTITUDE
```

`gps_go_to_goal_controller_node` consumes the goal vector, goal distance, and
vehicle yaw and publishes `/cmd_vel_navigation`. Defaults are intentionally
conservative: maximum 3.0 m/s (10.8 km/h), slowdown inside 12 m, and stop inside
3 m.
While `manual_control.py` is running, press `Y` to pause ROS control, position
the car manually at A, and press `Y` again to hand control back to ROS.

`lidar_emergency_stop_node` is the only node that publishes the final
`/cmd_vel`. It stops immediately when at least 20 non-ground returns are found
inside the 2.7 m-wide corridor up to 6 m ahead. It resumes only after the 8 m
corridor is clear for three consecutive LiDAR scans. Missing LiDAR or command
data also produces a stop. Returns closer than 2.5 m to the roof LiDAR are
excluded as the Lincoln's own hood/ego footprint.

Safety diagnostics are published on:

- `/safety/state`
- `/safety/emergency_stop`
- `/safety/nearest_obstacle_distance`
- `/safety/obstacle_points`

Every run writes a flushed timeline to:

```text
~/terrain_nav_data/logs/safety/run_YYYYMMDD_HHMMSS_microseconds/safety.csv
```

The emergency-stop layer itself only stops; steering detours are produced by
the local-avoidance layer below.

## Smooth Ackermann waypoint routes

The F7/F9 waypoint route is converted from WGS84 to the same local ENU frame
as the vehicle. Internal corners are rounded inside the waypoint polyline and
remain within 2 m of each waypoint, inside the default 3 m arrival radius.
The controller tracks `/navigation/smoothed_path` with forward-only Pure
Pursuit and limits curvature to the Lincoln MKZ wheelbase and steering-angle
parameters. Consequently, the vehicle follows a continuous curve instead of
aiming at each waypoint as an independent straight segment.

Route construction occurs once when START ROUTE is pressed. At 0.5 m path
spacing, normal routes contain only a few hundred points; each 10 Hz control
cycle searches a small forward window. The cost is negligible compared with
CARLA rendering and 3D LiDAR processing. In RViz, add a `Path` display for
`/navigation/smoothed_path` to inspect the exact reference curve.

## Single-obstacle local avoidance

`local_avoidance_node` sits between GNSS guidance and the emergency-stop gate:

```text
/cmd_vel_navigation
  -> local_avoidance_node
  -> /cmd_vel_avoidance
  -> lidar_emergency_stop_node
  -> /cmd_vel
```

The avoidance path is an odometry-anchored smooth S-curve tracked with pure
pursuit. Each left/right candidate checks the complete swept corridor using the
configured vehicle width plus lateral clearance. The pass phase is locked until
odometry confirms the requested lateral offset for repeated control cycles.
Downstream emergency-stop time is excluded from the active detour timeout.
Obstacles located beyond the remaining GNSS stopping point are ignored so they
do not trigger an unnecessary detour immediately before arrival.

It checks centre, left, and right corridors out to 17 m. When the centre is
blocked, it locks one open side and shifts to the configured clearance line.
It does not assume an obstacle length or a fixed passing distance. The roof
LiDAR instead anchors the initially detected obstacle in the odometry-aligned
detour frame and expands its longitudinal extent only with spatially connected
returns. This prevents unrelated buildings, poles, and kerbs farther along the
road from extending the pass phase. Once enough returns from the tracked object
cross behind the rear boundary, its measured extent is frozen so later city
clutter cannot be appended to it. The vehicle holds the offset until the frozen
rear extent moves behind the rear bumper plus the configured clearance and the
return corridor stays clear for three consecutive scans. It then
creates the smooth return portion and rejoins navigation guidance. For an
F7/F9 route, recovery uses the nearest forward segment and tangent on
`/navigation/smoothed_path`; a single F6 goal retains the straight goal-line
fallback. Pressing START ROUTE clears any unfinished avoidance state before
the new path begins, so an old recovery line cannot steer a new route. The
downstream 6 m emergency stop remains active throughout the manoeuvre. If both
sides are
blocked, the local node publishes zero velocity. A manoeuvre timeout also stops
the vehicle instead of returning without LiDAR passage confirmation.

Useful diagnostics are:

- `/avoidance/state`
- `/avoidance/selected_side`
- `/avoidance/center_points`
- `/avoidance/left_points`
- `/avoidance/right_points`
- `/avoidance/path`
- `/avoidance/actual_lateral_offset`

The avoidance CSV also records the current phase, points that have not passed
the rear boundary, points confirmed behind it, the consecutive-clear count,
the return-corridor point count, and the tracked obstacle's minimum/maximum
progress. These fields distinguish a genuine LiDAR pass confirmation from a
timeout or emergency stop.

Runs are logged to:

```text
~/terrain_nav_data/logs/avoidance/run_*/avoidance.csv
```

The launch starts avoidance by default. To test the same command chain without
modifying GNSS commands, use `avoidance_enabled:=false`; do not stop the local
node itself because the downstream safety gate intentionally treats a missing
command as unsafe.

Every run writes a flushed navigation timeline to:

```text
~/terrain_nav_data/logs/navigation/run_YYYYMMDD_HHMMSS_microseconds/
  navigation.csv
  run_metadata.json
```

The CSV includes raw GNSS, local ENU position, odometry, speed, destination,
distance, bearing, arrival state, sensor ages, and status. It is safe to
inspect while the node is running.

## Nav2 LiDAR obstacle and clearing clouds

`lidar_obstacle_filter_node` and the fail-safe authority node publish two
complementary paths:

- `/lidar/nav2_obstacles_baseline`: private, locally ground-fitted baseline
  obstacle returns.
- `/lidar/nav2_obstacles`: the selected marking cloud consumed by both Nav2
  and path clearance. In the default `baseline` mode it is an exact relay. In
  `add_only` mode it retains every baseline record and may append v2 obstacle
  endpoints; passable/unknown can never delete baseline points.
- `/lidar/nav2_clearing`: a sparse full-scan cloud used only for costmap
  raytracing/clearing. Separating these sources prevents a transient road
  false-positive from remaining in the costmap after it disappears from the
  obstacle-filtered cloud.

The local ground estimator fits a small plane around each XY grid cell. It
therefore follows road pitch and crown while retaining abrupt curb/step
returns. Returns consistent with that surface are removed even when an
uphill road rises above the legacy sensor-frame fixed-Z threshold.

## RViz navigation diagnostics

`navigation_visualization_node` is started by
`terrain_navigation.launch.py`. It does not modify any control command. It
translates the `map`-frame reference route into `odom` and records a
distance-sampled driven path for visualization:

- `/navigation/smoothed_path_odom`: blue global reference route
- `/avoidance/path`: red currently selected local trajectory
- `/navigation/driven_path`: white accumulated vehicle track
- `/lidar/points`: height-coloured live 3D LiDAR scan

Open the saved top-down layout in a second terminal:

```bash
ros2 launch launch_pkg terrain_navigation_rviz.launch.py
```

The RViz fixed frame is `odom`, and the camera follows `base_link`. Mouse-wheel
zoom can be used to switch between the local avoidance view and the complete
building route. A new F9 waypoint route clears the accumulated green path.

`F9 / OPEN ROUTE` drives the saved targets in the order
`WP1 -> WP2 -> ... -> WPn` and stops at the final waypoint. `F10 / CLOSED LAP`
treats WP1 as the start/finish marker and drives
`WP2 -> ... -> WPn -> WP1`, then stops after one lap. Both modes use the same
two-target rolling horizon, so the mapless local costmap only has to plan the
current target and one preview target instead of validating the entire distant
route at once. F10 deliberately does not use the route protocol's infinite
`loop` flag, so a closed-lap test cannot continue circling unattended.

For both rolling modes, a two-pose preview is used on long segments. When the
vehicle comes within `rolling_waypoint_focus_distance_m` (6 m by default) of
the current intermediate waypoint, the bridge temporarily replaces that
preview with a current-waypoint-only action. This prevents the 3.5 m RPP
lookahead from cutting the corner toward the following waypoint outside the
strict 1 m capture/crossing corridor. Passing the waypoint immediately starts
the next two-pose rolling window, so the long-range controller settings remain
unchanged.

If Nav2 aborts a two-pose rolling preview because the second, future waypoint
is not yet connected in the observed rolling costmap, the bridge automatically
retries the same current waypoint as a one-pose action. This prevents an
unobservable or temporarily blocked preview waypoint from stopping progress to
the reachable current waypoint. A one-pose retry is attempted only once for
that waypoint; failure of the retry remains a real navigation failure.

The committed avoidance side is not released merely because the centre
corridor is briefly clear. The vehicle must first achieve the configured
odometry-measured lateral shift (`trajectory_side_lock_minimum_lateral_m`).
This prevents an early turn back into the same obstacle.

## Nav2 long-obstacle preview mode

The legacy GNSS/Pure-Pursuit/local-trajectory chain remains the default. A
separate Nav2 launch is provided for building-scale planning so comparison and
rollback do not require editing the working launch file.

Preview first (Nav2 is physically disconnected from CARLA `/cmd_vel`):

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=false
```

With CARLA sensors and TF running, press F9 for an open rolling route or F10
for one closed rolling lap. Inspect `/global_costmap/costmap`, `/local_costmap/costmap`,
`/plan`, `/lidar/nav2_obstacles`, and `/navigation/nav2_goal` in the existing
terrain-navigation RViz configuration. The preview launch may publish
`/cmd_vel_nav2`, but no node forwards that topic to the vehicle.

Only after the costmaps and Smac path are visibly correct, enable driving:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true
```

Do not run `terrain_navigation.launch.py` at the same time. In Nav2 drive
mode, commands follow this explicit boundary:

```text
Smac Hybrid-A* -> MPPI Ackermann -> velocity smoother (/cmd_vel_nav2)
  -> lidar_emergency_stop_node -> CARLA (/cmd_vel)
```

`lidar_obstacle_filter_node` removes ground/sky returns in the roof-LiDAR
frame and voxel-downsamples the scan before both costmaps consume it. The
global costmap is a 100 x 100 m rolling online map, so it supports the current
container/building test and manually supplied waypoint corridors without a
pre-authored map. A destination farther than this planning window should be
split into intermediate GNSS waypoints. Persistent campus-scale mapping or a
rolling subgoal manager is intentionally a later stage.

The planner uses the Lincoln footprint, 0.35 m padding, and a 4.1 m minimum
turning radius. The custom behavior tree replans at 1 Hz but contains no
in-place Spin recovery, since that motion is impossible for the real car.

## FAR-inspired long-range guide mode

The original `direct` bridge remains the default so the previously verified
F9/F10 behavior is not silently changed. An experimental two-level mode is
available for routes whose current GNSS waypoint is difficult for Smac Hybrid
to solve as one long Ackermann search:

```text
F9/F10 GNSS waypoint
  -> online coarse occupancy-grid search
  -> line-of-sight visibility vertices (long guide)
  -> next 12 m subgoal with guide tangent
  -> Smac Hybrid-A* short Ackermann path
  -> RPP controller -> safety gates -> vehicle
```

This first implementation is **FAR-inspired**, not a claim that the official
FAR Planner repository has been copied verbatim. The official system expects
its own terrain-cloud/intensity contract. Here, the existing Nav2 rolling
costmap is used directly so the architecture can be tested without replacing
the verified LiDAR, TF, command, and safety interfaces. Unknown cells remain
searchable with a penalty; observed free space is preferred. Smac Hybrid and
the independent safety layers still have final authority over actual motion.

Build and test it with the same saved F7 waypoints:

```bash
cd ~/terrain_nav_ws
colcon build --symlink-install \
  --packages-select terrain_navigation_pkg config_pkg launch_pkg
source install/setup.bash

ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true \
  guide_mode:=far
```

Then press F9 for an open route or F10 for a closed lap. In RViz:

- green `/navigation/far_guide_path`: long online guide
- orange `/navigation/far_subgoal`: the current short Smac target
- magenta `/plan`: the actual kinematically feasible Smac Hybrid path

Useful diagnostics are:

```bash
ros2 topic echo /navigation/far_guide_status
ros2 topic echo /navigation/far_subgoal
```

The proposed stability profile keeps a still-hard-valid active Hybrid
segment sticky. A newly revealed rolling-costmap guide cannot cancel the
active action merely because it is shorter. A hard-blocked path also requires
a candidate subgoal at least `active_replan_subgoal_change_m` away
before FAR cancels the action; smaller guide jitter is left to the behavior
tree's commit-on-success replanning. A materially different blocked corridor
and Nav2 action failure still trigger FAR replanning. The former valid-path
optimization remains available for an ablation run:

The independent raw-LiDAR safety gate also feeds persistent commanded-arc
rejections back to FAR. Three distinct rejected scans within one second cancel
only the current short Nav2 action and temporarily penalize its guide corridor,
so an alternate can be searched without waiting for Nav2's 15-second progress
timeout. This feedback never clears obstacle data or releases the emergency
stop.

```bash
ros2 param set /far_nav2_guide_node \
  active_replan_allow_valid_path_optimization true
```

Restart the launch or set the parameter back to `false` before evaluating the
proposed profile.

Each run is logged under
`~/terrain_nav_data/logs/far_guide/run_*/far_guide.csv`. To return to the
previous behavior for regression comparison, use `guide_mode:=direct` or omit
the argument entirely.

The coarse guide treats OccupancyGrid values 99 and 100 as hard collision
space. Lower inflation values remain traversable soft costs. This preserves
the vehicle's inscribed collision boundary without sealing a usable corridor
merely because it lies inside Nav2's preferred-clearance halo.

## YAML mission routes

Routes measured beforehand can be written directly as WGS84 coordinates in a
YAML file. The loader reads and validates that file, converts it in memory to
the existing `/navigation/waypoint_route` JSON message, and publishes it. It
does not copy anything into Pygame's
`~/.config/terrain_navigation/waypoints.json`, so F7/F9/F10 and YAML missions
do not overwrite each other's saved files.

Copy the installed template or edit the source template:

```text
~/terrain_nav_ws/src/config_pkg/config/routes/mission_route_template.yaml
```

With the Nav2 launch already running, publish and start a mission in another
terminal:

```bash
ros2 launch launch_pkg terrain_mission_route.launch.py \
  route_file:=/absolute/path/to/my_route.yaml \
  start_route:=true
```

Alternatively, start Nav2 and the route loader together:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true \
  mission_route_enabled:=true \
  mission_route_file:=/absolute/path/to/my_route.yaml \
  mission_route_start:=true
```

`start_route:=false` only loads the route. `start_route:=true` overrides a
false `auto_start` value in YAML without editing the file. The loader remains
alive after its one latched publication so late subscribers can still receive
the route. `/navigation/mission_route_status` reports the loaded/published/
started state. Only one source should actively start a route at a time; a
later F9/F10 press intentionally replaces the currently loaded runtime route.

The `temporary_waypoints` block is reserved for automatic 25--30 m mission
subdivision. It is validated now but not yet applied to navigation.

## Sensor-only localization shadow

The simulation can run a `robot_localization` chain that mirrors the planned
VN-200 plus AKN940/AKA900 real-vehicle inputs without giving it navigation
authority. CARLA continues to publish `/vehicle/odometry` and the only
`odom -> base_link` transform. All shadow filters set `publish_tf: false`:

- `/localization/odometry_gps_shadow`: startup-relative AKN940/AKA900 ENU
  position with VN-200-based GNSS antenna lever-arm correction
- `/localization/gnss_shadow_status`: GNSS adapter initialization/freshness
- `/localization/odometry_shadow`: GNSS position fused with VN-200 yaw and
  yaw rate; acceleration-only dead reckoning is intentionally excluded until
  measured wheel velocity is available
- `/localization/shadow_status`: aligned error metrics against CARLA truth

The evaluator treats the two alignment contracts independently. GNSS ENU
position already has absolute east/north axes, so its arbitrary startup
translation is removed without rotating the x/y trajectory. VN-200 heading is
allowed a separate constant startup yaw alignment. Rotating the ENU positions
by that AHRS yaw correction would create an artificial error proportional to
distance travelled.

Each run now writes two files:

- `localization_shadow.csv`: fused EKF output against isolated CARLA truth
- `gnss_adapter.csv`: pre-EKF GNSS/lever-arm output against the same truth

This split distinguishes projection/extrinsic errors from errors introduced by
the filter. `/localization/shadow_status` also reports signed longitudinal and
lateral bias plus a `transition_readiness` object. `shadow_quality_ready` means
only that the configured metric thresholds were met. `control_trial_ready`
additionally requires an independently verified velocity source and exclusive
sensor ownership of the navigation `odom -> base_link` transform.

Start it with the normal stack:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=true \
  guide_mode:=far \
  localization_shadow_enabled:=true \
  gnss_projection_mode:=carla_mercator
```

`carla_mercator` is an explicit simulator compatibility mode used by both the
GNSS goal manager and the shadow-localization adapter. CARLA's legacy GNSS
georeference uses a Mercator planar scale that is measurably different from a
WGS84 ENU tangent plane. Do not pass this argument on the real vehicle: the
default `wgs84` mode is the production conversion for AKN940/AKA900 fixes.

`manual_control.py` publishes `/clock` from CARLA simulation time for these
three shadow nodes. Restart Pygame after updating the script. The evaluator
writes CSV files under
`~/terrain_nav_data/logs/localization_shadow/run_*/`. Ground truth is consumed
only by that evaluator and never by the GNSS adapter or global EKF.

Until a measured wheel encoder or another independent longitudinal-velocity
source is added, IMU acceleration is deliberately not integrated. The global
EKF infers horizontal velocity from successive noisy GNSS positions, so this
mode remains unsuitable for active navigation; its purpose is to quantify the
sensor-only gap before changing the control odometry.

Do not change only Nav2's `odom_topic` to `/localization/odometry_shadow`.
CARLA still owns the authoritative `odom -> base_link` TF, while the shadow
estimate lives in `odom_sensor_shadow` and publishes no TF. Mixing sensor twist
with the ground-truth pose would look like a sensor-mode test while retaining
ground-truth localization. The active transition needs all of the following as
one controlled change:

1. a measured wheel encoder or another non-ground-truth longitudinal velocity,
2. exactly one sensor-owned navigation TF publisher,
3. CARLA ground-truth TF disabled while `/vehicle/odometry` remains available
   only to the isolated evaluator,
4. Nav2, the GNSS goal manager, FAR, validators, safety recorder, and costmaps
   configured to one consistent sensor odometry frame,
5. a stationary preflight where `transition_readiness.control_trial_ready`
   is true before drive authority is enabled.

## Rolling 3D terrain mapper

`terrain_mapping_node` transforms every complete `/lidar/points` revolution
into `odom`, removes returns inside the ego-vehicle footprint, and fuses
repeated observations in a bounded 3D voxel map. It publishes:

- `/terrain/map_points`: persistent fused 3D voxels
- `/terrain/slope_costmap`: slope magnitude encoded from 0 to 100
- `/terrain/traversability_costmap`: combined slope and step cost

Both costmaps use `-1` for unknown cells. A cost of 0 is locally flat, and 100
meets or exceeds either the configured lethal slope or lethal step threshold.
The default map retains voxels within 40 m of the vehicle, rather than using
RViz's temporary PointCloud2 decay buffer.

Run the mapper by itself:

```bash
ros2 run terrain_navigation_pkg terrain_mapping_node \
  --ros-args \
  --params-file ~/terrain_nav_ws/src/config_pkg/config/params.yaml
```

In RViz, keep `Fixed Frame` at `odom`, set `/terrain/map_points` to
`Decay Time = 0`, and add both OccupancyGrid topics. The node itself owns the
map lifetime, so RViz must only display the latest published map.
