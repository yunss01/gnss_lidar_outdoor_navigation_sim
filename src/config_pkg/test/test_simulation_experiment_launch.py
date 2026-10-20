import importlib.util
from pathlib import Path

import yaml


def _module():
    path = Path(__file__).resolve(
    ).parents[3] / 'src/launch_pkg/launch/simulation_experiment.launch.py'
    spec = importlib.util.spec_from_file_location('experiment_launch_test',
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_nested_costmaps_get_simulation_clock(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'config'
    module = _module()
    common = yaml.safe_load((root / 'params.yaml').read_text())
    nav2 = yaml.safe_load((root / 'nav2_outdoor_params.yaml').read_text())
    module._rewrite(common, tmp_path)
    module._rewrite(nav2, tmp_path)
    assert nav2['controller_server']['ros__parameters']['use_sim_time']
    assert nav2['local_costmap']['local_costmap']['ros__parameters'][
        'use_sim_time']
    assert nav2['global_costmap']['global_costmap']['ros__parameters'][
        'use_sim_time']
    for name in ('gnss_goal_manager_node', 'lidar_emergency_stop_node',
                 'far_nav2_guide_node', 'path_clearance_validator_node'):
        assert common[name]['ros__parameters']['use_sim_time']
        assert common[name]['ros__parameters']['log_directory'] == str(
            tmp_path / name)
    # Read-only source files were not changed by the rewrite.
    assert not yaml.safe_load(
        (root / 'nav2_outdoor_params.yaml'
         ).read_text())['controller_server']['ros__parameters']['use_sim_time']


def test_experiment_launch_refuses_wrong_domain(monkeypatch):
    module = _module()
    monkeypatch.setenv('ROS_DOMAIN_ID', '42')
    import pytest
    with pytest.raises(RuntimeError, match='domain 113'):
        module._launch(None)
