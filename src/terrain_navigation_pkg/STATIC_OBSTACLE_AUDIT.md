# No-Authority Static-Obstacle Audit Protocol

## Purpose

This protocol checks the passive learned/baseline obstacle-cloud candidate
before it is allowed to affect Nav2. The candidate remains an RViz and audit
topic throughout this test. The established rule-based cloud continues to
control the Nav2 costmaps and LiDAR safety gate.

The test answers four questions:

1. Does learned-free ever remove a semantically verified obstacle that the
   baseline preserved?
2. Does learned-obstacle add useful obstacle cells missed by the baseline?
3. Does the candidate add excessive ghost occupancy in verified free space?
4. Is exact-timestamp fusion produced at a stable rate without gaining any
   navigation or braking authority?

This is a simulation integration gate. Passing it is not proof of real-world
safety or unseen-map generalization.

## Controlled single-prop placement

When a naturally isolated low obstacle is unavailable, park the CARLA
operator vehicle on an empty, level surface. Preview the point 6 m ahead
without changing the world:

```bash
ros2 run terrain_navigation_pkg controlled_carla_obstacle preview \
  --forward-m 6.0 --right-m 0.0
```

Spawn one low box only after checking the previewed surface:

```bash
ros2 run terrain_navigation_pkg controlled_carla_obstacle spawn \
  --blueprint static.prop.creasedbox01 \
  --forward-m 6.0 --right-m 0.0
```

The tool records only the spawned actor ID. Inspect or remove that exact actor
with:

```bash
ros2 run terrain_navigation_pkg controlled_carla_obstacle status
ros2 run terrain_navigation_pkg controlled_carla_obstacle remove
```

It refuses to spawn a second controlled prop while the recorded actor remains
active. Removal also refuses unexpected non-`static.prop.*` actor types.
The tool joins the running synchronous simulation with `wait_for_tick()` and
never advances the world itself; `manual_control.py` must remain open as the
single tick owner.

Some runtime-spawned props are reported by CARLA semantic LiDAR with tag 0
(Unknown), although their `object_idx` is valid. Pass the actor ID printed by
the spawn tool to the audit so that only that controlled actor is treated as
obstacle ground truth:

```bash
ros2 run terrain_navigation_pkg traversability_static_obstacle_audit_node \
  --ros-args -p controlled_obstacle_actor_id:=ACTOR_ID
```

## Required topics

```text
/lidar/nav2_obstacles
/learning/nav2_obstacles_candidate
/learning/obstacle_candidate_status
/lidar/semantic_points
/lidar/points
/learning/traversability_probability_float
/learning/traversability_entropy_float
/learning/traversability_mc_variance_float
/learning/traversability_hard_obstacle_mask
/learning/traversability_shadow_decision
```

The semantic LiDAR topic is privileged evaluation input only. It is never an
input to the deployed model or vehicle controller.

## Safety boundary

- Use `drive_enabled:=false`.
- Do not change either Nav2 costmap to the candidate topic.
- Do not change the LiDAR safety gate to the candidate topic.
- The candidate topic may have only RViz, the audit recorder, and temporary
  diagnostic subscribers.
- The status must report both `navigation_control_effect=none` and
  `safety_control_effect=none`.

## Test scene

Keep the vehicle stationary. Place one clear static object in front of the
vehicle and record its type, approximate distance from the vehicle front, and
lateral placement.

Supported labels should follow this pattern:

```text
wall_center_5m
vehicle_left_5m
container_right_5m
low_curb_center_3m
```

Use vehicle-forward and vehicle-left coordinates:

- `center`: object intersects the projected driving corridor;
- `left`: object is offset to the vehicle-left side;
- `right`: object is offset to the vehicle-right side.

## Staged scenario matrix

Do not collect the full matrix until the first recorder smoke test succeeds.

### Stage A: recorder smoke test

| Scenario | Purpose |
|---|---|
| wall_center_5m | Verify topics, alignment, output files, and no-authority status |

### Stage B: required obstacle coverage

For each object type below, collect center, left, and right placement at the
nominal distance. Also collect one nearer center placement.

| Object type | Nominal distance | Near center distance |
|---|---:|---:|
| wall/building face | 5 m | 3 m |
| parked vehicle | 5 m | 3 m |
| container | 5 m | 3 m |
| low curb/step | 3 m | 2 m |

This produces 16 required captures: four placements per object type.

### Stage C: optional range sensitivity

After Stage B passes, add centered objects at approximately 8 m and 12 m.
These trials measure sparse-return sensitivity but are not a substitute for a
new-map test.

## CARLA and ROS startup

Start CARLA and the modified manual-control client with semantic LiDAR labels.
Use the normal map and spawn options for the current scene. The required flag
is:

```text
--semantic-lidar-labels
```

Build and source the workspace after changing the audit code:

```bash
cd /home/sukja/terrain_nav_ws
colcon build --symlink-install \
  --packages-select terrain_navigation_pkg launch_pkg
source /home/sukja/terrain_nav_ws/install/setup.bash
```

Start the navigation stack with AI shadow inference and the passive candidate,
but without vehicle actuation or route recorders:

```bash
ros2 launch launch_pkg terrain_navigation_nav2.launch.py \
  drive_enabled:=false \
  guide_mode:=far \
  record_learning_data:=false \
  traversability_shadow_enabled:=true \
  traversability_shadow_recording_enabled:=false \
  traversability_shadow_evaluation_enabled:=false \
  traversability_shadow_diagnostic_images_enabled:=true \
  traversability_candidate_enabled:=true
```

RViz may be started separately with the existing RViz launch. The green
display is `AI fused obstacle candidate (NO AUTHORITY)` and the established
baseline obstacle cloud remains the comparison source.

## Pre-capture checks

Verify that the required topics exist:

```bash
ros2 topic list | rg 'semantic_points|nav2_obstacles_candidate|obstacle_candidate_status|traversability_.*float|hard_obstacle_mask'
```

Check the candidate rate for several seconds:

```bash
ros2 topic hz /learning/nav2_obstacles_candidate
```

Inspect candidate subscribers:

```bash
ros2 topic info /learning/nav2_obstacles_candidate --verbose
```

Nav2 and the safety gate must not appear as subscribers.

## Capture one scenario

Open a new ROS 2 terminal and source the workspace:

```bash
source /home/sukja/terrain_nav_ws/install/setup.bash
```

Run a 20-second capture. Replace the example parameters with the actual scene.

```bash
ros2 run terrain_navigation_pkg \
  traversability_static_obstacle_audit_node --ros-args \
  -p scenario_label:=wall_center_5m \
  -p obstacle_type:=wall \
  -p obstacle_distance_m:=5.0 \
  -p obstacle_lateral_position:=center \
  -p map_name:=Town10HD_Opt \
  -p duration_s:=20.0 \
  -p warmup_s:=3.0
```

The node exits automatically. Do not move the vehicle or obstacle during the
20-second capture. A new directory is written below:

```text
/home/sukja/terrain_nav_data/learning/static_obstacle_audits
```

Each capture contains:

```text
metadata.json
frames.csv
instances.csv
summary.json
snapshots/representative.npz
snapshots/worst_removed_obstacle.npz   # only if such an error occurred
snapshots/worst_added_free.npz         # only if such an error occurred
```

The three selected snapshots, rather than every frame, additionally preserve
raw LiDAR XYZ, semantic XYZ/tag/object ID, LiDAR BEV height channels, exact
model probability, entropy, MC variance, the hard-obstacle mask, and the
selective decision. This bounds storage while retaining enough evidence to
distinguish model error, sparse LiDAR sampling, hard-mask behavior, and
cell-alignment effects.

`instances.csv` reports each CARLA obstacle `object_idx` intersecting the
forward diagnostic corridor. It records baseline/candidate cell support,
retention, and whether the candidate completely removed an instance that the
baseline detected. Semantic instance IDs remain privileged audit data and are
not supplied to the deployed model.

Render a reusable 300-DPI spatial diagnostic for the newest audit with:

```bash
ros2 run terrain_navigation_pkg visualize_static_obstacle_audit
```

The output is written below the selected audit's `diagnostics/` directory.
It separates added and removed cells by semantic obstacle, verified free, and
unknown labels and reports radial range bands plus the forward planning
corridor. An unknown cell means that the synchronized semantic scan supplied
no cell label; the plot alone cannot prove whether that cell is a physical
obstacle or a ghost return.

During the passive rollout, AI additions and clearings are provisionally
limited to a 20 m radial range. Beyond 20 m, the candidate retains the
established baseline cloud unchanged. This conservative boundary prevents
sparse far-field predictions from changing navigation evidence before
range-stratified validation is available.

## Automatic metrics

The recorder compares baseline and candidate occupancy against synchronized
CARLA semantic targets.

- `removed_target_obstacle`: semantic obstacle cells present in baseline but
  removed from the candidate; this is the primary unsafe-clearing count.
- `removed_target_free`: baseline ghost cells correctly removed by candidate.
- `added_target_obstacle`: semantic obstacle cells newly recovered by candidate.
- `added_target_free`: free cells newly occupied by candidate.
- `candidate_to_baseline_occupied_ratio`: candidate occupancy amplification.
- `candidate_rate_hz`: matched exact-timestamp candidate rate.
- `semantic_alignment.maximum_s`: maximum target alignment error.
- `status_authority_violations`: any status frame claiming control authority.
- `diagnostic_attached_fraction`: fraction of frames with all exact-stamp v2
  diagnostic arrays attached.
- `instance_evaluation.fully_removed_instance_observations`: repeated object
  observations for which baseline support was nonzero and candidate support
  became zero in the forward corridor.

Counts are repeated cell-observations across frames, not counts of independent
physical obstacles.

## Provisional automatic gate

The first passive gate requires:

- at least 40 matched frames;
- at least 4.0 Hz matched candidate rate;
- semantic timestamp alignment no greater than 30 ms;
- `removed_target_obstacle == 0`;
- maximum candidate/baseline occupied-cell ratio no greater than 1.25;
- candidate status attached to at least 95% of matched frames;
- all v2 diagnostics attached to at least 95% of matched frames;
- at least one baseline-detected obstacle instance in the diagnostic corridor;
- zero completely removed obstacle-instance observations;
- zero navigation or safety authority violations;
- zero candidate-cache eviction.

These numerical thresholds are provisional engineering checks. In particular,
the 1.25 amplification threshold is intended to reject the already observed
1.74x over-expansion while allowing small boundary differences. It is not a
certified safety limit.

## Manual review

Even if `summary.json` reports `pass: true`, review the following:

1. In RViz, the candidate still outlines the physical obstacle.
2. Removed cells lie on obvious free-ground false positives, not on the object.
3. Added cells correspond to the object or a plausible hard boundary.
4. Candidate occupancy does not form a large wall across open free space.
5. The vehicle remained stationary and no route command was issued.

If a scenario fails, preserve the entire audit directory. Do not delete or
replace the failed result. Use its worst-case NPZ snapshot to diagnose the
model, target builder, or fusion policy.

## Decision after the audit

Only after the required Stage B scenarios pass should a baseline-fallback mux
be implemented. The next authority stage is Nav2-only at low speed. The
independent rule-based LiDAR safety gate must retain final braking authority.
