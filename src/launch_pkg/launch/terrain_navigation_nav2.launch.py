"""Preview or drive mapless outdoor navigation with Nav2 and 3D LiDAR."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


EVALUATION_PROFILES = {
    'proposed': 'evaluation_proposed.yaml',
    'b0_fixed_ground': 'evaluation_b0_fixed_ground.yaml',
    'b1_plane_ground': 'evaluation_b1_plane_ground.yaml',
}


def _select_evaluation_profile(context):
    variant = LaunchConfiguration('evaluation_variant').perform(context)
    profile = EVALUATION_PROFILES.get(variant)
    if profile is None:
        raise RuntimeError(
            'evaluation_variant must be one of: {}'.format(
                ', '.join(EVALUATION_PROFILES)
            )
        )
    return [
        SetLaunchConfiguration('evaluation_profile_file', profile),
        LogInfo(msg='ICCE evaluation variant: {} ({})'.format(
            variant, profile
        )),
    ]


def generate_launch_description():
    config_share = FindPackageShare('config_pkg')
    common_parameters = PathJoinSubstitution([
        config_share,
        'config',
        'params.yaml',
    ])
    nav2_parameters = PathJoinSubstitution([
        config_share,
        'config',
        'nav2_outdoor_params.yaml',
    ])
    localization_shadow_parameters = PathJoinSubstitution([
        config_share,
        'config',
        'localization_shadow.yaml',
    ])
    evaluation_parameters = PathJoinSubstitution([
        config_share,
        'config',
        LaunchConfiguration('evaluation_profile_file'),
    ])
    rolling_behavior_tree = PathJoinSubstitution([
        config_share,
        'config',
        'nav2_ackermann_replanning.xml',
    ])
    nav_to_pose_behavior_tree = PathJoinSubstitution([
        config_share,
        'config',
        'nav2_ackermann_replanning.xml',
    ])
    nav_through_poses_behavior_tree = PathJoinSubstitution([
        config_share,
        'config',
        'nav2_ackermann_through_poses_replanning.xml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'drive_enabled',
            default_value='false',
            description=(
                'Pass Nav2 commands through the independent LiDAR safety gate'
            ),
        ),
        DeclareLaunchArgument(
            'guide_mode',
            default_value='direct',
            description=(
                "'direct' keeps the proven F9/F10 bridge; 'far' uses an "
                'online long-range guide and short Smac Hybrid segments'
            ),
        ),
        DeclareLaunchArgument('goal_enabled', default_value='false'),
        DeclareLaunchArgument('goal_latitude', default_value='0.0'),
        DeclareLaunchArgument('goal_longitude', default_value='0.0'),
        DeclareLaunchArgument('goal_altitude', default_value='0.0'),
        DeclareLaunchArgument(
            'gnss_projection_mode',
            default_value='wgs84',
            description=(
                'GNSS projection: wgs84 for real hardware; explicitly use '
                'carla_mercator only with the CARLA GNSS actor'
            ),
        ),
        DeclareLaunchArgument(
            'start_navigation_visualization',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'mission_route_enabled',
            default_value='false',
            description='Load a YAML mission route with this Nav2 stack',
        ),
        DeclareLaunchArgument(
            'mission_route_file',
            default_value='',
            description='Absolute path to a WGS84 mission-route YAML file',
        ),
        DeclareLaunchArgument(
            'mission_route_start',
            default_value='false',
            description='Start the YAML route after loading it',
        ),
        DeclareLaunchArgument(
            'record_learning_data',
            default_value='true',
            description=(
                'Record subscriber-only LiDAR/navigation training samples '
                'while F9/F10 is active'
            ),
        ),
        DeclareLaunchArgument(
            'collect_traversability_data',
            default_value='false',
            description=(
                'Save raw geometric LiDAR, training-only CARLA semantic '
                'labels, and IMU state during active F9/F10 routes'
            ),
        ),
        DeclareLaunchArgument(
            'controlled_traversability_actor_id',
            default_value='-1',
            description=(
                'CARLA semantic object_idx for one controlled actor; -1 '
                'disables actor-specific vehicle-policy supervision'
            ),
        ),
        DeclareLaunchArgument(
            'controlled_traversability_disposition',
            default_value='',
            description='Vehicle policy: obstacle, passable, or ambiguous',
        ),
        DeclareLaunchArgument(
            'controlled_traversability_blueprint',
            default_value='',
            description='Auditable CARLA blueprint of the controlled actor',
        ),
        DeclareLaunchArgument(
            'controlled_traversability_policy_source',
            default_value='vehicle_clearance_policy',
            description='Origin of the controlled actor disposition',
        ),
        DeclareLaunchArgument(
            'perception_capture',
            default_value='false',
            description=(
                'Record one stationary raw-LiDAR/costmap capture without '
                'starting F9/F10, then close it automatically.'
            ),
        ),
        DeclareLaunchArgument(
            'perception_capture_duration_s',
            default_value='15.0',
            description='Duration of the stationary perception capture.',
        ),
        DeclareLaunchArgument(
            'perception_capture_label',
            default_value='',
            description='Short scene label stored in capture metadata.',
        ),
        DeclareLaunchArgument(
            'perception_capture_output_directory',
            default_value='~/terrain_nav_data/perception_validation/raw',
            description='Directory for stationary perception captures.',
        ),
        DeclareLaunchArgument(
            'traversability_shadow_enabled',
            default_value='false',
            description=(
                'Run the learned traversability model on isolated display '
                'topics without navigation or safety authority'
            ),
        ),
        DeclareLaunchArgument(
            'localization_shadow_enabled',
            default_value='false',
            description=(
                'Run VN-200 plus GNSS robot_localization output in isolation; '
                'CARLA ground-truth odometry remains the navigation source'
            ),
        ),
        DeclareLaunchArgument(
            'localization_shadow_evaluation_enabled',
            default_value='true',
            description=(
                'Compare shadow odometry with CARLA truth in an evaluator '
                'that has no connection back to either EKF'
            ),
        ),
        DeclareLaunchArgument(
            'localization_shadow_output_directory',
            default_value='~/terrain_nav_data/logs/localization_shadow',
            description='Directory for sensor-shadow localization metrics.',
        ),
        DeclareLaunchArgument(
            'traversability_shadow_checkpoint',
            default_value=(
                '/home/sukja/terrain_nav_data/learning/models/'
                'traversability_pilot/v2_domain_aug/best.pt'
            ),
            description='Checkpoint used by the shadow traversability node.',
        ),
        DeclareLaunchArgument(
            'traversability_shadow_device',
            default_value='auto',
            description='Shadow inference device: auto, cpu, or cuda.',
        ),
        DeclareLaunchArgument(
            'traversability_shadow_mc_samples',
            default_value='4',
            description='Monte Carlo dropout samples per shadow inference.',
        ),
        DeclareLaunchArgument(
            'traversability_shadow_diagnostic_images_enabled',
            default_value='false',
            description=(
                'Publish exact float probability, entropy, MC variance, and '
                'hard-mask images for short passive audits only'
            ),
        ),
        DeclareLaunchArgument(
            'traversability_shadow_recording_enabled',
            default_value='true',
            description=(
                'Record route-level shadow metrics and sparse diagnostic '
                'grid snapshots when shadow inference is enabled'
            ),
        ),
        DeclareLaunchArgument(
            'traversability_shadow_output_directory',
            default_value='~/terrain_nav_data/learning/shadow_runs',
            description='Directory for compact shadow route recordings.',
        ),
        DeclareLaunchArgument(
            'traversability_shadow_evaluation_enabled',
            default_value='false',
            description=(
                'Evaluate shadow maps against CARLA semantic LiDAR. The '
                'semantic topic is privileged evaluation input only.'
            ),
        ),
        DeclareLaunchArgument(
            'traversability_shadow_evaluation_output_directory',
            default_value='~/terrain_nav_data/learning/shadow_evaluations',
            description='Directory for online semantic-evaluation results.',
        ),
        DeclareLaunchArgument(
            'traversability_evidence_v2_enabled',
            default_value='false',
            description=(
                'Run the current-only 8-channel v2 evidence model on '
                'isolated topics with no navigation authority'
            ),
        ),
        DeclareLaunchArgument(
            'traversability_evidence_v2_checkpoint',
            default_value=(
                '/home/sukja/terrain_nav_data/learning/models/'
                'traversability_evidence_v2/'
                'cv_temporal_representation_current_only_'
                'scenes09_11_20260921/'
                'holdout_scene11/model/best.pt'
            ),
            description=(
                'Current-only v2 evidence checkpoint used by the online '
                'shadow node'
            ),
        ),
        DeclareLaunchArgument(
            'traversability_evidence_v2_device',
            default_value='auto',
            description='Evidence v2 inference device: auto, cpu, or cuda.',
        ),
        DeclareLaunchArgument(
            'traversability_evidence_v2_mc_samples',
            default_value='1',
            description=(
                'MC dropout samples; one preserves deterministic offline '
                'evaluation parity'
            ),
        ),
        DeclareLaunchArgument(
            'traversability_evidence_v2_diagnostics_enabled',
            default_value='false',
            description='Publish float v2 probability/variance images.',
        ),
        DeclareLaunchArgument(
            'traversability_obstacle_authority_mode',
            default_value='baseline',
            description=(
                'Selected Nav2 cloud policy: baseline, passive, or add_only. '
                'No mode may delete baseline obstacles.'
            ),
        ),
        DeclareLaunchArgument(
            'traversability_candidate_enabled',
            default_value='true',
            description=(
                'Publish a passive AI/baseline obstacle-cloud candidate when '
                'shadow inference is enabled. The candidate is not consumed '
                'by Nav2 or the safety gate.'
            ),
        ),
        DeclareLaunchArgument(
            'traversability_candidate_maximum_range_m',
            default_value='20.0',
            description=(
                'Maximum radius where passive AI decisions may add or clear '
                'candidate obstacle points; baseline is retained beyond it.'
            ),
        ),
        DeclareLaunchArgument(
            'evaluation_variant',
            default_value='proposed',
            description=(
                'Ground-processing profile: proposed, b0_fixed_ground, or '
                'b1_plane_ground'
            ),
        ),
        OpaqueFunction(function=_select_evaluation_profile),
        Node(
            package='terrain_navigation_pkg',
            executable='gnss_odometry_shadow_node',
            name='gnss_odometry_shadow_node',
            output='screen',
            parameters=[
                localization_shadow_parameters,
                {
                    'projection_mode': LaunchConfiguration(
                        'gnss_projection_mode'
                    ),
                },
            ],
            condition=IfCondition(
                LaunchConfiguration('localization_shadow_enabled')
            ),
        ),
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_sensor_global_shadow',
            output='screen',
            parameters=[localization_shadow_parameters],
            remappings=[
                ('odometry/filtered', '/localization/odometry_shadow'),
            ],
            condition=IfCondition(
                LaunchConfiguration('localization_shadow_enabled')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='localization_shadow_evaluator_node',
            name='localization_shadow_evaluator_node',
            output='screen',
            parameters=[
                localization_shadow_parameters,
                {
                    'output_directory': LaunchConfiguration(
                        'localization_shadow_output_directory'
                    ),
                },
            ],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('localization_shadow_enabled'),
                "' == 'true' and '",
                LaunchConfiguration(
                    'localization_shadow_evaluation_enabled'
                ),
                "' == 'true'",
            ])),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='gnss_goal_manager_node',
            name='gnss_goal_manager_node',
            output='screen',
            parameters=[
                common_parameters,
                {
                    'goal_enabled': ParameterValue(
                        LaunchConfiguration('goal_enabled'),
                        value_type=bool,
                    ),
                    'goal_latitude': ParameterValue(
                        LaunchConfiguration('goal_latitude'),
                        value_type=float,
                    ),
                    'goal_longitude': ParameterValue(
                        LaunchConfiguration('goal_longitude'),
                        value_type=float,
                    ),
                    'goal_altitude': ParameterValue(
                        LaunchConfiguration('goal_altitude'),
                        value_type=float,
                    ),
                    'projection_mode': LaunchConfiguration(
                        'gnss_projection_mode'
                    ),
                },
            ],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='gnss_waypoint_manager_node',
            name='gnss_waypoint_manager_node',
            output='screen',
            parameters=[
                common_parameters,
                {
                    'require_nav2_success_for_final_completion': (
                        ParameterValue(
                            PythonExpression([
                                "'",
                                LaunchConfiguration('guide_mode'),
                                "' != 'far'",
                            ]),
                            value_type=bool,
                        )
                    ),
                },
            ],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='lidar_obstacle_filter_node',
            name='lidar_obstacle_filter_node',
            output='screen',
            parameters=[common_parameters, evaluation_parameters],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='traversability_obstacle_authority_node',
            name='traversability_obstacle_authority_node',
            output='screen',
            parameters=[common_parameters, {
                'authority_mode': LaunchConfiguration(
                    'traversability_obstacle_authority_mode'
                ),
            }],
            respawn=True,
            respawn_delay=1.0,
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='nav2_goal_bridge_node',
            name='nav2_goal_bridge_node',
            output='screen',
            parameters=[
                common_parameters,
                {'rolling_behavior_tree': rolling_behavior_tree},
            ],
            condition=IfCondition(PythonExpression([
                "'",
                LaunchConfiguration('guide_mode'),
                "' == 'direct'",
            ])),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='far_nav2_guide_node',
            name='far_nav2_guide_node',
            output='screen',
            parameters=[
                common_parameters,
                {'behavior_tree': rolling_behavior_tree},
            ],
            condition=IfCondition(PythonExpression([
                "'",
                LaunchConfiguration('guide_mode'),
                "' == 'far'",
            ])),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='path_clearance_validator_node',
            name='path_clearance_validator_node',
            output='screen',
            parameters=[common_parameters],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='path_validity_gate_node',
            name='path_validity_gate_node',
            output='screen',
            parameters=[common_parameters],
            condition=IfCondition(LaunchConfiguration('drive_enabled')),
        ),
        # Explicit Nav2 processes are used instead of including the standard
        # launch file. This makes the command boundary unambiguous:
        # controller -> /nav2/cmd_vel_raw -> smoother -> /cmd_vel_nav2.
        # Nothing reaches CARLA's /cmd_vel unless drive_enabled starts the
        # independent emergency-stop node below.
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[nav2_parameters],
            remappings=[('cmd_vel', '/nav2/cmd_vel_raw')],
        ),
        Node(
            package='nav2_smoother',
            executable='smoother_server',
            name='smoother_server',
            output='screen',
            parameters=[nav2_parameters],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[nav2_parameters],
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[nav2_parameters],
            remappings=[('cmd_vel', '/nav2/behavior_cmd_vel')],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[
                nav2_parameters,
                {
                    'default_nav_to_pose_bt_xml': nav_to_pose_behavior_tree,
                    'default_nav_through_poses_bt_xml': (
                        nav_through_poses_behavior_tree
                    ),
                },
            ],
        ),
        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen',
            parameters=[nav2_parameters],
        ),
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            output='screen',
            parameters=[nav2_parameters],
            remappings=[
                ('cmd_vel', '/nav2/cmd_vel_raw'),
                ('cmd_vel_smoothed', '/cmd_vel_nav2'),
            ],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': [
                    'controller_server',
                    'smoother_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator',
                    'waypoint_follower',
                    'velocity_smoother',
                ],
            }],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='lidar_emergency_stop_node',
            name='lidar_emergency_stop_node',
            output='screen',
            parameters=[
                common_parameters,
                evaluation_parameters,
                {
                    'input_command_topic': '/cmd_vel_path_validated',
                    'output_command_topic': '/cmd_vel',
                },
            ],
            condition=IfCondition(LaunchConfiguration('drive_enabled')),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='navigation_visualization_node',
            name='navigation_visualization_node',
            output='screen',
            parameters=[common_parameters],
            condition=IfCondition(
                LaunchConfiguration('start_navigation_visualization')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='mission_route_loader_node',
            name='mission_route_loader_node',
            output='screen',
            parameters=[{
                'route_file': LaunchConfiguration('mission_route_file'),
                'start_route': ParameterValue(
                    LaunchConfiguration('mission_route_start'),
                    value_type=bool,
                ),
            }],
            condition=IfCondition(
                LaunchConfiguration('mission_route_enabled')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='traversability_shadow_node',
            name='traversability_shadow_node',
            output='screen',
            parameters=[{
                'checkpoint_path': LaunchConfiguration(
                    'traversability_shadow_checkpoint'
                ),
                'device': LaunchConfiguration(
                    'traversability_shadow_device'
                ),
                'mc_samples': ParameterValue(
                    LaunchConfiguration('traversability_shadow_mc_samples'),
                    value_type=int,
                ),
                'publish_diagnostic_images': ParameterValue(
                    LaunchConfiguration(
                        'traversability_shadow_diagnostic_images_enabled'
                    ),
                    value_type=bool,
                ),
            }],
            condition=IfCondition(
                LaunchConfiguration('traversability_shadow_enabled')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='traversability_shadow_recorder_node',
            name='traversability_shadow_recorder_node',
            output='screen',
            parameters=[{
                'output_directory': LaunchConfiguration(
                    'traversability_shadow_output_directory'
                ),
            }],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('traversability_shadow_enabled'),
                "' == 'true' and '",
                LaunchConfiguration(
                    'traversability_shadow_recording_enabled'
                ),
                "' == 'true'",
            ])),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='traversability_obstacle_candidate_node',
            name='traversability_obstacle_candidate_node',
            output='screen',
            parameters=[common_parameters, {
                'maximum_range_m': ParameterValue(
                    LaunchConfiguration(
                        'traversability_candidate_maximum_range_m'
                    ),
                    value_type=float,
                ),
            }],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('traversability_shadow_enabled'),
                "' == 'true' and '",
                LaunchConfiguration('traversability_candidate_enabled'),
                "' == 'true'",
            ])),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='traversability_evidence_shadow_node',
            name='traversability_evidence_shadow_node',
            output='screen',
            parameters=[common_parameters, {
                'checkpoint_path': LaunchConfiguration(
                    'traversability_evidence_v2_checkpoint'
                ),
                'device': LaunchConfiguration(
                    'traversability_evidence_v2_device'
                ),
                'mc_samples': ParameterValue(
                    LaunchConfiguration(
                        'traversability_evidence_v2_mc_samples'
                    ),
                    value_type=int,
                ),
                'publish_diagnostic_images': ParameterValue(
                    LaunchConfiguration(
                        'traversability_evidence_v2_diagnostics_enabled'
                    ),
                    value_type=bool,
                ),
            }],
            condition=IfCondition(
                LaunchConfiguration('traversability_evidence_v2_enabled')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='traversability_shadow_evaluator_node',
            name='traversability_shadow_evaluator_node',
            output='screen',
            parameters=[{
                'output_directory': LaunchConfiguration(
                    'traversability_shadow_evaluation_output_directory'
                ),
            }],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('traversability_shadow_enabled'),
                "' == 'true' and '",
                LaunchConfiguration(
                    'traversability_shadow_evaluation_enabled'
                ),
                "' == 'true'",
            ])),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='navigation_learning_recorder_node',
            name='navigation_learning_recorder_node',
            output='screen',
            parameters=[
                common_parameters,
                {
                    'evaluation_variant': LaunchConfiguration(
                        'evaluation_variant'
                    ),
                    'perception_capture_on_start': ParameterValue(
                        LaunchConfiguration('perception_capture'),
                        value_type=bool,
                    ),
                    'perception_capture_duration_s': ParameterValue(
                        LaunchConfiguration('perception_capture_duration_s'),
                        value_type=float,
                    ),
                    'perception_capture_label': LaunchConfiguration(
                        'perception_capture_label'
                    ),
                    'perception_capture_output_directory': LaunchConfiguration(
                        'perception_capture_output_directory'
                    ),
                    # Keep normal F9/F10 evaluation lightweight.  Capture
                    # mode alone saves the raw cloud and BEV/costmap arrays.
                    'save_sample_files': ParameterValue(
                        PythonExpression([
                            "'", LaunchConfiguration('perception_capture'),
                            "' == 'true' or '",
                            LaunchConfiguration('collect_traversability_data'),
                            "' == 'true'",
                        ]),
                        value_type=bool,
                    ),
                    'save_raw_points': ParameterValue(
                        PythonExpression([
                            "'", LaunchConfiguration('perception_capture'),
                            "' == 'true' or '",
                            LaunchConfiguration('collect_traversability_data'),
                            "' == 'true'",
                        ]),
                        value_type=bool,
                    ),
                    'save_semantic_labels': ParameterValue(
                        LaunchConfiguration('collect_traversability_data'),
                        value_type=bool,
                    ),
                    'controlled_traversability_actor_id': ParameterValue(
                        LaunchConfiguration(
                            'controlled_traversability_actor_id'
                        ),
                        value_type=int,
                    ),
                    'controlled_traversability_disposition': (
                        LaunchConfiguration(
                            'controlled_traversability_disposition'
                        )
                    ),
                    'controlled_traversability_blueprint': (
                        LaunchConfiguration(
                            'controlled_traversability_blueprint'
                        )
                    ),
                    'controlled_traversability_policy_source': (
                        LaunchConfiguration(
                            'controlled_traversability_policy_source'
                        )
                    ),
                },
            ],
            condition=IfCondition(
                LaunchConfiguration('record_learning_data')
            ),
        ),
    ])
