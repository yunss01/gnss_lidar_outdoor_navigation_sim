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


def _launch(context):
    if os.environ.get('ROS_DOMAIN_ID') != '113':
        raise RuntimeError(
            'use scripts/experiment.sh; experiment requires ROS domain 113')
    config = Path(get_package_share_directory('config_pkg')) / 'config'
    launch_file = Path(get_package_share_directory('launch_pkg')) / 'launch'
    log_root = Path(
        LaunchConfiguration('experiment_log_root').perform(
            context)).expanduser()
    temporary = []
    paths = []
    for name in ('params.yaml', 'nav2_outdoor_params.yaml'):
        payload = yaml.safe_load((config / name).read_text())
        _rewrite(payload, log_root)
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
        LogInfo(
            msg='SIM EXPERIMENT: domain 113, simulation clock, separate logs'),
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
            'experiment_log_root',
            default_value='~/terrain_nav_data/experiments/node_logs'),
        OpaqueFunction(function=_launch),
    ])
