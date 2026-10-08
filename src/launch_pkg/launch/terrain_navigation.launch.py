"""Start the first terrain-navigation components."""

from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    parameters = PathJoinSubstitution([
        FindPackageShare('config_pkg'),
        'config',
        'params.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'goal_enabled',
            default_value='false',
            description='Load the destination from launch arguments',
        ),
        DeclareLaunchArgument(
            'goal_latitude',
            default_value='0.0',
        ),
        DeclareLaunchArgument(
            'goal_longitude',
            default_value='0.0',
        ),
        DeclareLaunchArgument(
            'goal_altitude',
            default_value='0.0',
        ),
        DeclareLaunchArgument(
            'gnss_projection_mode',
            default_value='wgs84',
            description=(
                'GNSS projection: wgs84 for real hardware; explicitly use '
                'carla_mercator only with the CARLA GNSS actor'
            ),
        ),
        DeclareLaunchArgument(
            'start_terrain_mapping',
            default_value='false',
        ),
        DeclareLaunchArgument(
            'start_gps_controller',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'start_lidar_emergency_stop',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'start_local_avoidance',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'avoidance_enabled',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'start_navigation_visualization',
            default_value='true',
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='terrain_mapping_node',
            name='terrain_mapping_node',
            output='screen',
            parameters=[parameters],
            condition=IfCondition(
                LaunchConfiguration('start_terrain_mapping')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='gnss_goal_manager_node',
            name='gnss_goal_manager_node',
            output='screen',
            parameters=[
                parameters,
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
            parameters=[parameters],
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='gps_go_to_goal_controller_node',
            name='gps_go_to_goal_controller_node',
            output='screen',
            parameters=[parameters],
            condition=IfCondition(
                LaunchConfiguration('start_gps_controller')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='lidar_emergency_stop_node',
            name='lidar_emergency_stop_node',
            output='screen',
            parameters=[parameters],
            condition=IfCondition(
                LaunchConfiguration('start_lidar_emergency_stop')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='local_avoidance_node',
            name='local_avoidance_node',
            output='screen',
            parameters=[
                parameters,
                {
                    'avoidance_enabled': ParameterValue(
                        LaunchConfiguration('avoidance_enabled'),
                        value_type=bool,
                    ),
                },
            ],
            condition=IfCondition(
                LaunchConfiguration('start_local_avoidance')
            ),
        ),
        Node(
            package='terrain_navigation_pkg',
            executable='navigation_visualization_node',
            name='navigation_visualization_node',
            output='screen',
            parameters=[parameters],
            condition=IfCondition(
                LaunchConfiguration('start_navigation_visualization')
            ),
        ),
    ])
