"""Start the Ouster OS0 and expose its cloud on the terrain LiDAR API."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Connect to the lab OS0 without starting navigation or vehicle control."""
    sensor_launch = PathJoinSubstitution([
        FindPackageShare('ouster_ros'),
        'launch',
        'sensor.launch.xml',
    ])

    points_topic = LaunchConfiguration('points_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'sensor_hostname',
            default_value='192.168.100.1',
            description='Static IPv4 address of the Ouster OS0.',
        ),
        DeclareLaunchArgument(
            'udp_dest',
            default_value='192.168.100.100',
            description='IPv4 address of the host OS0 Ethernet interface.',
        ),
        DeclareLaunchArgument(
            'points_topic',
            default_value='/lidar/points',
            description='Terrain-navigation PointCloud2 input topic.',
        ),
        DeclareLaunchArgument(
            'lidar_mode',
            default_value='1024x10',
            description='OS0 horizontal resolution and rotation rate.',
        ),
        DeclareLaunchArgument(
            'timestamp_mode',
            default_value='TIME_FROM_INTERNAL_OSC',
            description='OS0 timestamp source used before GNSS time sync.',
        ),
        DeclareLaunchArgument(
            'viz',
            default_value='false',
            description='Start the Ouster RViz configuration.',
        ),
        GroupAction([
            SetRemap(src='/ouster/points', dst=points_topic),
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(sensor_launch),
                launch_arguments={
                    'sensor_hostname': LaunchConfiguration(
                        'sensor_hostname'
                    ),
                    'udp_dest': LaunchConfiguration('udp_dest'),
                    'lidar_mode': LaunchConfiguration('lidar_mode'),
                    'timestamp_mode': LaunchConfiguration('timestamp_mode'),
                    'point_type': 'xyzi',
                    'proc_mask': 'PCL|IMU',
                    'attempt_reconnect': 'true',
                    'viz': LaunchConfiguration('viz'),
                }.items(),
            ),
        ]),
    ])
