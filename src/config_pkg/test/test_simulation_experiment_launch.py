import importlib.util
from pathlib import Path

import yaml
import pytest


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


@pytest.mark.parametrize('radius', [1.0, 2.0, 3.0])
def test_arrival_radius_only_changes_temporary_gnss_config(radius):
    root = Path(__file__).resolve().parents[1] / 'config'
    path = root / 'params.yaml'
    original = path.read_bytes()
    payload = yaml.safe_load(original)
    expected = yaml.safe_load(original)
    expected['gnss_goal_manager_node']['ros__parameters'][
        'arrival_radius_m'] = radius
    _module()._set_arrival_radius(payload, radius)
    assert payload == expected
    assert path.read_bytes() == original
    assert yaml.safe_load(original)['gnss_goal_manager_node'][
        'ros__parameters']['arrival_radius_m'] == 3.0
    nav2 = yaml.safe_load((root / 'nav2_outdoor_params.yaml').read_text())
    assert nav2['controller_server']['ros__parameters'][
        'general_goal_checker']['xy_goal_tolerance'] == 1.0


@pytest.mark.parametrize('radius', ['nan', 'inf', '0.5', '3.01', '-1', 'bad'])
def test_launch_rejects_invalid_radius_before_temp_files(monkeypatch, radius):
    module = _module()
    monkeypatch.setenv('ROS_DOMAIN_ID', '113')
    monkeypatch.setattr(
        module, 'LaunchConfiguration',
        lambda _name: type('Value', (), {
            'perform': lambda _self, _context: radius
        })())
    monkeypatch.setattr(
        module.tempfile, 'NamedTemporaryFile',
        lambda **_kwargs: pytest.fail('must validate before writes'))
    with pytest.raises(ValueError):
        module._launch(None)


@pytest.mark.parametrize('variant,distance', [
    ('baseline', 0.0), ('rejoin', 8.0),
])
def test_navigation_variant_changes_temporary_far_only(variant, distance):
    root = Path(__file__).resolve().parents[1] / 'config'
    path = root / 'params.yaml'
    original = path.read_bytes()
    payload = yaml.safe_load(original)
    _module()._set_navigation_variant(payload, variant)
    parameters = payload['far_nav2_guide_node']['ros__parameters']
    assert parameters['terminal_approach_distance_m'] == distance
    assert parameters['terminal_approach_capture_m'] == 1.0
    assert path.read_bytes() == original
    assert 'terminal_approach_distance_m' not in yaml.safe_load(original)[
        'far_nav2_guide_node']['ros__parameters']
