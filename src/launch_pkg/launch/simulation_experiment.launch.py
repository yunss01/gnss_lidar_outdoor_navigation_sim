"""Isolated experiment stack; standard launch defaults stay unchanged."""

from pathlib import Path
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from terrain_navigation_pkg.simulation_experiment_core import (
    navigation_variant_parameters,
    validate_arrival_radius,
)
from terrain_navigation_pkg.simulation_path_trace import new_log_session
import yaml


def _rewrite(tree, log_root, trail=()):
    for name, value in tree.items():
        if not isinstance(value, dict):
            continue
        if name == 'ros__parameters':
            value['use_sim_time'] = True
            if 'log_directory' in value:
                value['log_directory'] = str(log_root.joinpath(*trail))
        else:
            _rewrite(value, log_root, trail + (name, ))


def _set_arrival_radius(payload, value):
    radius = validate_arrival_radius(value)
    payload['gnss_goal_manager_node']['ros__parameters'][
        'arrival_radius_m'] = radius


def _set_navigation_variant(payload, variant):
    payload['far_nav2_guide_node']['ros__parameters'].update(
        navigation_variant_parameters(variant))


def _launch(context):
    if os.environ.get('ROS_DOMAIN_ID') != '113':
        raise RuntimeError(
            'use scripts/experiment.sh; experiment requires ROS domain 113')
    radius = validate_arrival_radius(
        LaunchConfiguration('arrival_radius_m').perform(context))
    variant = LaunchConfiguration('navigation_variant').perform(context)
    navigation_variant_parameters(variant)
    config = Path(get_package_share_directory('config_pkg')) / 'config'
    launch_file = Path(get_package_share_directory('launch_pkg')) / 'launch'
    log_root = Path(
        LaunchConfiguration('experiment_log_root').perform(
            context)).expanduser()
    # A validator may start before /clock arrives. Isolate every launch even
    # when a node's own timestamp-based child directory is called run_0.
    log_root = new_log_session(log_root)
    temporary = []
    paths = []
    for name in ('params.yaml', 'nav2_outdoor_params.yaml'):
        payload = yaml.safe_load((config / name).read_text())
        _rewrite(payload, log_root)
        if name == 'params.yaml':
            _set_arrival_radius(payload, radius)
            _set_navigation_variant(payload, variant)
        with tempfile.NamedTemporaryFile(mode='w',
                                         prefix='terrain-experiment-',
                                         suffix='.yaml',
                                         delete=False) as stream:
            yaml.safe_dump(payload, stream)
            paths.append(stream.name)
            temporary.append(Path(stream.name))

    def cleanup(_context):
        for path in temporary:
            path.unlink(missing_ok=True)
        return []

    return [
        RegisterEventHandler(
            OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])),
        LogInfo(msg=(
            'SIM EXPERIMENT: domain 113, simulation clock, separate logs; '
            f'GNSS arrival_radius_m={radius:.1f}; variant={variant} '
            f'(experiment only); node logs={log_root}')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(launch_file / 'terrain_navigation_nav2.launch.py')),
                                 launch_arguments={
                                     'common_parameters_file':
                                     paths[0],
                                     'nav2_parameters_file':
                                     paths[1],
                                     'navigation_use_sim_time':
                                     'true',
                                     'gnss_projection_mode':
                                     'carla_mercator',
                                     'drive_enabled':
                                     'true',
                                     'guide_mode':
                                     'far',
                                     'record_learning_data':
                                     'false',
                                     'start_navigation_visualization':
                                     LaunchConfiguration('viz'),
                                     'traversability_shadow_enabled':
                                     'false',
                                     'traversability_evidence_v2_enabled':
                                     'false',
                                     'traversability_obstacle_authority_mode':
                                     'baseline',
                                     'localization_shadow_enabled':
                                     'false',
                                 }.items()),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('viz', default_value='true'),
        DeclareLaunchArgument(
            'navigation_variant',
            default_value='baseline',
            description='baseline or rejoin (8 m before goal, same safety)'),
        DeclareLaunchArgument(
            'arrival_radius_m',
            default_value='3.0',
            description=('Experiment GNSS arrival radius, 1.0 to 3.0 m; '
                         'scoring unchanged')),
        DeclareLaunchArgument(
            'experiment_log_root',
            default_value='~/terrain_nav_data/experiments/node_logs'),
        OpaqueFunction(function=_launch),
    ])
