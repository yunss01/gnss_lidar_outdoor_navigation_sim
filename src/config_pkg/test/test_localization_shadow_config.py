from pathlib import Path

import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / 'config'
    / 'localization_shadow.yaml'
)
LAUNCH_PATH = (
    Path(__file__).resolve().parents[2]
    / 'launch_pkg'
    / 'launch'
    / 'terrain_navigation_nav2.launch.py'
)


def _parameters(node_name):
    document = yaml.safe_load(CONFIG_PATH.read_text())
    return document[node_name]['ros__parameters']


def test_shadow_filter_cannot_publish_conflicting_tf():
    parameters = _parameters('ekf_sensor_global_shadow')
    assert parameters['publish_tf'] is False
    assert parameters['odom_frame'] == 'odom_sensor_shadow'


def test_shadow_filter_inputs_do_not_include_ground_truth_odometry():
    document = yaml.safe_load(CONFIG_PATH.read_text())
    serialized = yaml.safe_dump({
        name: document[name]
        for name in (
            'gnss_odometry_shadow_node',
            'ekf_sensor_global_shadow',
        )
    })

    assert '/vehicle/odometry' not in serialized
    assert _parameters('gnss_odometry_shadow_node')['imu_topic'] == (
        '/vectornav/imu'
    )
    assert _parameters('ekf_sensor_global_shadow')['odom0'] == (
        '/localization/odometry_gps_shadow'
    )
    assert 'odom1' not in _parameters('ekf_sensor_global_shadow')


def test_ground_truth_is_confined_to_evaluator_configuration():
    assert _parameters(
        'localization_shadow_evaluator_node')['ground_truth_topic'] == (
            '/vehicle/odometry'
        )


def test_shadow_nodes_share_carla_simulation_time():
    for node_name in (
            'gnss_odometry_shadow_node',
            'ekf_sensor_global_shadow'):
        assert _parameters(node_name)['use_sim_time'] is True


def test_real_vehicle_projection_is_default_and_launch_override_is_shared():
    assert _parameters('gnss_odometry_shadow_node')['projection_mode'] == (
        'wgs84'
    )
    source = LAUNCH_PATH.read_text(encoding='utf-8')
    shared_projection = (
        "LaunchConfiguration(\n"
        "                        'gnss_projection_mode'"
    )

    assert "'gnss_projection_mode'" in source
    assert source.count(shared_projection) >= 2
    assert 'carla_mercator only with the CARLA GNSS actor' in source


def test_shadow_launch_uses_tf_independent_gnss_adapter():
    """The shadow path must not depend on CARLA's ground-truth TF tree."""
    source = LAUNCH_PATH.read_text(encoding='utf-8')

    assert "executable='gnss_odometry_shadow_node'" in source
    assert "executable='navsat_transform_node'" not in source


def test_unobservable_imu_only_velocity_is_not_fused():
    parameters = _parameters('ekf_sensor_global_shadow')

    assert '/localization/odometry_local_shadow' not in yaml.safe_dump(
        parameters
    )
    assert parameters['imu0_config'][12:15] == [False, False, False]


def test_transition_gate_cannot_enable_sensor_control_yet():
    parameters = _parameters('localization_shadow_evaluator_node')

    assert parameters['active_navigation_tf_ready'] is False
    assert parameters['independent_velocity_ready'] is False
    assert parameters['transition_maximum_position_p95_m'] == 0.30
    assert parameters['transition_maximum_longitudinal_p95_m'] == 0.25
    assert parameters['transition_maximum_lateral_p95_m'] == 0.20
    assert (
        parameters['transition_maximum_absolute_longitudinal_bias_m']
        == 0.10
    )
    assert (
        parameters['transition_maximum_relative_translation_p95_m']
        == 0.08
    )


def test_shadow_evaluator_receives_shared_localization_parameters():
    source = LAUNCH_PATH.read_text(encoding='utf-8')
    evaluator_start = source.index(
        "executable='localization_shadow_evaluator_node'")
    evaluator_block = source[evaluator_start:evaluator_start + 700]

    assert 'localization_shadow_parameters' in evaluator_block


def test_shadow_evaluator_tracks_narrow_passage_metrics():
    source = (
        Path(__file__).resolve().parents[2]
        / 'terrain_navigation_pkg'
        / 'terrain_navigation_pkg'
        / 'localization_shadow_evaluator_node.py'
    ).read_text(encoding='utf-8')

    assert "'lateral_error_m'" in source
    assert "'relative_translation_error_m'" in source
    assert "'pose_jump_count'" in source
