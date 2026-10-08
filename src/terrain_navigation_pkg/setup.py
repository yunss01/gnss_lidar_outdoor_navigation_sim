from setuptools import find_packages, setup


package_name = 'terrain_navigation_pkg'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            [
                'package.xml', 'README.md', 'PLANNING_DESIGN.md',
                'EVALUATION.md', 'STATIC_OBSTACLE_AUDIT.md',
                'TRAVERSABILITY_EVIDENCE_V2.md',
            ],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sukja',
    maintainer_email='sukja@todo.todo',
    description='GNSS-guided terrain-aware navigation nodes',
    license='GPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gnss_goal_manager_node = '
            'terrain_navigation_pkg.gnss_goal_manager_node:main',
            'gnss_waypoint_manager_node = '
            'terrain_navigation_pkg.gnss_waypoint_manager_node:main',
            'terrain_mapping_node = '
            'terrain_navigation_pkg.terrain_mapping_node:main',
            'gps_go_to_goal_controller_node = '
            'terrain_navigation_pkg.gps_go_to_goal_controller_node:main',
            'lidar_emergency_stop_node = '
            'terrain_navigation_pkg.lidar_emergency_stop_node:main',
            'local_avoidance_node = '
            'terrain_navigation_pkg.local_avoidance_node:main',
            'navigation_visualization_node = '
            'terrain_navigation_pkg.navigation_visualization_node:main',
            'nav2_goal_bridge_node = '
            'terrain_navigation_pkg.nav2_goal_bridge_node:main',
            'lidar_obstacle_filter_node = '
            'terrain_navigation_pkg.lidar_obstacle_filter_node:main',
            'path_clearance_validator_node = '
            'terrain_navigation_pkg.path_clearance_validator_node:main',
            'path_validity_gate_node = '
            'terrain_navigation_pkg.path_validity_gate_node:main',
            'mission_route_loader_node = '
            'terrain_navigation_pkg.mission_route_loader_node:main',
            'far_nav2_guide_node = '
            'terrain_navigation_pkg.far_nav2_guide_node:main',
            'localization_shadow_evaluator_node = '
            'terrain_navigation_pkg.'
            'localization_shadow_evaluator_node:main',
            'gnss_odometry_shadow_node = '
            'terrain_navigation_pkg.gnss_odometry_shadow_node:main',
            'navigation_learning_recorder_node = '
            'terrain_navigation_pkg.navigation_learning_recorder_node:main',
            'evaluate_navigation_runs = '
            'terrain_navigation_pkg.evaluate_navigation_runs:main',
            'build_learning_manifest = '
            'terrain_navigation_pkg.navigation_learning_manifest:main',
            'build_traversability_dataset = '
            'terrain_navigation_pkg.build_traversability_dataset:main',
            'build_traversability_evidence_dataset = '
            'terrain_navigation_pkg.'
            'build_traversability_evidence_dataset:main',
            'build_traversability_temporal_evidence_dataset = '
            'terrain_navigation_pkg.'
            'build_traversability_temporal_evidence_dataset:main',
            'prepare_traversability_evidence_experiment = '
            'terrain_navigation_pkg.'
            'prepare_traversability_evidence_experiment:main',
            'prepare_traversability_temporal_experiment = '
            'terrain_navigation_pkg.'
            'prepare_traversability_temporal_experiment:main',
            'visualize_traversability_dataset = '
            'terrain_navigation_pkg.visualize_traversability_dataset:main',
            'visualize_traversability_evidence_dataset = '
            'terrain_navigation_pkg.'
            'visualize_traversability_evidence_dataset:main',
            'train_traversability_model = '
            'terrain_navigation_pkg.train_traversability_model:main',
            'train_traversability_evidence_model = '
            'terrain_navigation_pkg.'
            'train_traversability_evidence_model:main',
            'evaluate_traversability_model = '
            'terrain_navigation_pkg.evaluate_traversability_model:main',
            'evaluate_traversability_evidence_model = '
            'terrain_navigation_pkg.'
            'evaluate_traversability_evidence_model:main',
            'evaluate_traversability_temporal_pilot = '
            'terrain_navigation_pkg.'
            'evaluate_traversability_temporal_pilot:main',
            'cross_validate_traversability_evidence_model = '
            'terrain_navigation_pkg.'
            'cross_validate_traversability_evidence_model:main',
            'visualize_traversability_predictions = '
            'terrain_navigation_pkg.'
            'visualize_traversability_predictions:main',
            'traversability_shadow_node = '
            'terrain_navigation_pkg.traversability_shadow_node:main',
            'traversability_evidence_shadow_node = '
            'terrain_navigation_pkg.'
            'traversability_evidence_shadow_node:main',
            'validate_traversability_evidence_online_parity = '
            'terrain_navigation_pkg.'
            'validate_traversability_evidence_online_parity:main',
            'traversability_shadow_recorder_node = '
            'terrain_navigation_pkg.traversability_shadow_recorder_node:main',
            'traversability_shadow_evaluator_node = '
            'terrain_navigation_pkg.traversability_shadow_evaluator_node:main',
            'traversability_obstacle_candidate_node = '
            'terrain_navigation_pkg.'
            'traversability_obstacle_candidate_node:main',
            'traversability_obstacle_authority_node = '
            'terrain_navigation_pkg.'
            'traversability_obstacle_authority_node:main',
            'traversability_static_obstacle_audit_node = '
            'terrain_navigation_pkg.'
            'traversability_static_obstacle_audit_node:main',
            'visualize_static_obstacle_audit = '
            'terrain_navigation_pkg.'
            'visualize_static_obstacle_audit:main',
            'controlled_carla_obstacle = '
            'terrain_navigation_pkg.controlled_carla_obstacle:main',
            'stress_test_traversability_model = '
            'terrain_navigation_pkg.stress_test_traversability_model:main',
            'visualize_learning_trajectories = '
            'terrain_navigation_pkg.visualize_navigation_learning:main',
            'train_navigation_trajectory_model = '
            'terrain_navigation_pkg.train_navigation_trajectory_model:main',
            'evaluate_navigation_trajectory_model = '
            'terrain_navigation_pkg.evaluate_navigation_trajectory_model:main',
            'visualize_navigation_trajectory_predictions = '
            'terrain_navigation_pkg.'
            'visualize_navigation_trajectory_predictions:main',
        ],
    },
)
