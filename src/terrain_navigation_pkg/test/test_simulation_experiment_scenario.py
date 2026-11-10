import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from terrain_navigation_pkg import simulation_experiment_scenario as scenario
from terrain_navigation_pkg.simulation_experiment import _parser, report
from terrain_navigation_pkg.simulation_experiment import navigation_fingerprint
from terrain_navigation_pkg.simulation_experiment import _preflight_missing
from terrain_navigation_pkg.simulation_experiment import _verify_arrival_radius
from terrain_navigation_pkg import simulation_experiment as experiment


def test_connection_waits_for_actor_snapshot_without_ticking(monkeypatch):
    hero = SimpleNamespace(id=7, attributes={'role_name': 'hero'})
    visible = []
    calls = []

    def next_snapshot(seconds):
        calls.append(seconds)
        visible.append(hero)
        return object()

    def forbidden_tick():
        pytest.fail('observer must not advance the world')

    world = SimpleNamespace(
        get_actors=lambda: SimpleNamespace(filter=lambda _pattern: visible),
        wait_for_tick=next_snapshot,
        tick=forbidden_tick)
    client = SimpleNamespace(set_timeout=lambda _seconds: None,
                             get_world=lambda: world)
    carla = SimpleNamespace(Client=lambda _host, _port: client)
    monkeypatch.setattr(scenario, 'import_carla', lambda _root: carla)
    options = SimpleNamespace(carla_root='/tmp/carla',
                              host='127.0.0.1',
                              port=2000)
    assert not world.get_actors().filter('vehicle.*')
    assert scenario.connect(options) == (carla, world)
    assert scenario._select_vehicle(world) is hero
    assert calls == [3.0]


def test_connection_timeout_explains_missing_client(monkeypatch):

    def no_snapshot(_seconds):
        raise RuntimeError('time-out waiting for simulator')

    world = SimpleNamespace(wait_for_tick=no_snapshot)
    client = SimpleNamespace(set_timeout=lambda _seconds: None,
                             get_world=lambda: world)
    monkeypatch.setattr(
        scenario, 'import_carla',
        lambda _root: SimpleNamespace(Client=lambda _host, _port: client))
    options = SimpleNamespace(carla_root='/tmp/carla',
                              host='127.0.0.1',
                              port=2000)
    with pytest.raises(RuntimeError, match='Keep terminal B'):
        scenario.connect(options)


def test_preflight_does_not_wait_for_position_that_requires_a_goal(
        monkeypatch):
    monkeypatch.setattr(experiment.time, 'monotonic', lambda: 100.0)
    node = SimpleNamespace(bridge={
        'profile': 'baseline',
        'armed': False
    },
                           origin=object(),
                           current_local=None,
                           truth={1: object()},
                           nav={1: object()},
                           safety_state='clear',
                           lidar_wall=99.9,
                           bridge_wall=99.9)
    options = SimpleNamespace(profile='baseline')
    assert _preflight_missing(node, options) == []
    node.lidar_wall = 99.0
    assert _preflight_missing(node, options) == ['fresh_lidar']


def test_obstacle_placement_waits_for_new_actor_and_final_pose():
    target = SimpleNamespace(location=SimpleNamespace(x=-106.6,
                                                      y=-24.4,
                                                      z=-0.05),
                             rotation=SimpleNamespace(yaw=-91.46))
    zero = SimpleNamespace(location=SimpleNamespace(x=0, y=0, z=0),
                           rotation=SimpleNamespace(yaw=0))
    pose = [zero]
    waits = []

    def next_snapshot(seconds):
        waits.append(seconds)
        if len(waits) == 1:
            return SimpleNamespace(find=lambda _actor_id: None)
        if len(waits) == 3:
            pose[0] = target
        return SimpleNamespace(find=lambda _actor_id: object())

    actor = SimpleNamespace(id=43, get_transform=lambda: pose[0])
    world = SimpleNamespace(wait_for_tick=next_snapshot)
    assert scenario._wait_for_placement(world, actor, target) is target
    assert len(waits) == 3


def test_unconfirmed_obstacle_pose_cannot_be_saved(monkeypatch):
    timestamps = iter(range(10))
    monkeypatch.setattr(scenario.time, 'monotonic', lambda: next(timestamps))
    zero = SimpleNamespace(location=SimpleNamespace(x=0, y=0, z=0),
                           rotation=SimpleNamespace(yaw=0))
    target = SimpleNamespace(location=SimpleNamespace(x=20, y=0, z=0),
                             rotation=SimpleNamespace(yaw=0))
    actor = SimpleNamespace(id=43, get_transform=lambda: zero)
    world = SimpleNamespace(wait_for_tick=lambda _seconds: SimpleNamespace(
        find=lambda _actor_id: object()))
    with pytest.raises(RuntimeError, match='placement was not confirmed'):
        scenario._wait_for_placement(world, actor, target)


def test_anchor_is_exclusive_and_never_overwrites_user_file(tmp_path):
    path = tmp_path / 'anchor.json'
    scenario.write_json(path, {'original': 1}, exclusive=True)
    with pytest.raises(FileExistsError):
        scenario.write_json(path, {'replacement': 2}, exclusive=True)
    assert json.loads(path.read_text()) == {'original': 1}


def test_gnss_origin_is_attached_sensor_not_vehicle_body():
    hero = SimpleNamespace(id=7)
    location = SimpleNamespace(x=11.0, y=5.0, z=2.0)
    sensor = SimpleNamespace(
        parent=hero, get_transform=lambda: SimpleNamespace(location=location))
    wrong = SimpleNamespace(parent=SimpleNamespace(id=9))
    world = SimpleNamespace(get_actors=lambda: SimpleNamespace(
        filter=lambda _pattern: [sensor, wrong]))
    assert scenario.gnss_measurement_origin(world, hero) is location


def test_cleanup_validates_all_identities_before_any_destroy(
        tmp_path, monkeypatch):
    removed = []
    state = {
        'map_name':
        'Town',
        'hero_actor_id':
        1,
        'role_name':
        'owned',
        'obstacles': [
            dict(actor_id=2, blueprint='vehicle.test'),
            dict(actor_id=3, blueprint='vehicle.test')
        ],
    }
    scenario.write_json(tmp_path / 'prepared.json', state)
    actors = {
        2:
        SimpleNamespace(type_id='vehicle.test',
                        attributes={'role_name': 'owned'},
                        destroy=lambda: removed.append(2)),
        3:
        SimpleNamespace(type_id='vehicle.test',
                        attributes={'role_name': 'USER_CAR'},
                        destroy=lambda: removed.append(3)),
    }
    world = SimpleNamespace(get_map=lambda: SimpleNamespace(name='Town'),
                            get_actor=lambda actor_id: actors.get(actor_id))
    monkeypatch.setattr(scenario, 'connect', lambda _args: (None, world))
    with pytest.raises(RuntimeError, match='identity mismatch'):
        scenario.cleanup(SimpleNamespace(anchor=str(tmp_path / 'anchor.json')))
    assert removed == []


def test_report_retains_failure_and_uses_separate_unique_outputs(tmp_path):
    for index, successful in enumerate((True, False)):
        directory = tmp_path / f'run_{index}'
        directory.mkdir()
        scenario.write_json(
            directory / 'summary.json', {
                'run_id': directory.name,
                'profile': 'baseline',
                'scenario': 'S1',
                'success': successful,
                'metrics_valid': True,
            })
        scenario.write_json(directory / 'metadata.json',
                            {'anchor_sha256': 'abc'})
    report(SimpleNamespace(output_root=str(tmp_path)))
    files = list(tmp_path.glob('report_*.csv'))
    assert len(files) == 1
    assert len(files[0].read_text().splitlines()) == 3
    assert 'False' in files[0].read_text()
    report(SimpleNamespace(output_root=str(tmp_path)))
    assert len(list(tmp_path.glob('report_*.csv'))) == 2


def test_cli_has_all_operational_commands():
    parser = _parser()
    assert parser.parse_args(['find-start']).command == 'find-start'
    options = parser.parse_args([
        'run', '--anchor', '/tmp/anchor.json', '--scenario', 'S1', '--profile',
        'steering_both'
    ])
    assert options.max_sim_seconds == 120
    assert options.profile == 'steering_both'
    assert options.arrival_radius_m == 3.0
    assert parser.parse_args([
        'run', '--anchor', 'anchor.json', '--scenario', 'S1',
        '--arrival-radius-m', '1.0'
    ]).arrival_radius_m == 1.0


@pytest.mark.parametrize('radius',
                         ['nan', 'inf', '-1', '0', '0.9', '3.1', 'bad'])
def test_cli_rejects_invalid_arrival_radius(radius):
    with pytest.raises(SystemExit):
        _parser().parse_args([
            'run', '--anchor', 'anchor.json', '--scenario', 'S1',
            '--arrival-radius-m', radius
        ])


@pytest.mark.parametrize('case', [
    'ok', 'mismatch', 'missing', 'wrong_type', 'nan', 'out_of_range',
    'unavailable', 'timeout'
])
def test_active_radius_is_read_only_verified_and_client_is_cleaned(case):
    from rcl_interfaces.msg import ParameterValue
    value = ParameterValue(type=3, double_value=3.0)
    if case == 'wrong_type':
        value.type = 4
    if case == 'nan':
        value.double_value = float('nan')
    if case == 'out_of_range':
        value.double_value = 4.0
    requests, destroyed = [], []
    response = SimpleNamespace(values=[] if case == 'missing' else [value])
    future = SimpleNamespace(done=lambda: case != 'timeout',
                             result=lambda: response)

    def call(request):
        requests.append(request.names)
        return future

    client = SimpleNamespace(
        wait_for_service=lambda **_kw: case != 'unavailable', call_async=call)
    services = []

    def create(_kind, name):
        services.append(name)
        return client

    node = SimpleNamespace(create_client=create,
                           destroy_client=destroyed.append,
                           active_arrival_radius_m=None)
    ros = SimpleNamespace(
        spin_until_future_complete=lambda *_args, **_kw: None)
    requested = 1.0 if case == 'mismatch' else 3.0
    if case == 'ok':
        assert _verify_arrival_radius(node, requested, ros) == 3.0
    else:
        with pytest.raises(RuntimeError, match='preflight:'):
            _verify_arrival_radius(node, requested, ros)
    assert services == ['/gnss_goal_manager_node/get_parameters']
    assert destroyed == [client]
    assert requests == ([]
                        if case == 'unavailable' else [['arrival_radius_m']])
    if case in ('ok', 'mismatch'):
        assert node.active_arrival_radius_m == 3.0


def test_report_separates_active_radius_and_does_not_guess_legacy(
        tmp_path, capsys):
    configurations = [({}, 'legacy'),
                      ({
                          'requested_arrival_radius_m': 3.0,
                          'active_arrival_radius_m': 3.0,
                          'arrival_radius_verified': True
                      }, 'three'),
                      ({
                          'requested_arrival_radius_m': 1.0,
                          'active_arrival_radius_m': 1.0,
                          'arrival_radius_verified': True
                      }, 'one'),
                      ({
                          'requested_arrival_radius_m': 1.0,
                          'active_arrival_radius_m': 3.0,
                          'arrival_radius_verified': False
                      }, 'mismatch')]
    for configuration, name in configurations:
        directory = tmp_path / f'run_{name}'
        directory.mkdir()
        scenario.write_json(
            directory / 'summary.json',
            dict(run_id=directory.name,
                 scenario='S1',
                 profile='baseline',
                 success=False,
                 metrics_valid=False))
        scenario.write_json(
            directory / 'metadata.json',
            dict(anchor_sha256='abc',
                 navigation_code_sha256='same',
                 **configuration))
    report(SimpleNamespace(output_root=str(tmp_path)))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5  # CSV announcement + four distinct groups.
    assert any('requested=None active=None verified=False' in line
               for line in lines)
    assert any('requested=3.0 active=3.0 verified=True' in line
               for line in lines)
    assert any('requested=1.0 active=1.0 verified=True' in line
               for line in lines)
    assert any('requested=1.0 active=3.0 verified=False' in line
               for line in lines)
    with next(tmp_path.glob('report_*.csv')).open() as stream:
        rows = {row['run_id']: row for row in csv.DictReader(stream)}
    assert rows['run_legacy']['active_arrival_radius_m'] == ''
    assert rows['run_one']['active_arrival_radius_m'] == '1.0'
    assert len(rows) == 4


def test_source_manifest_has_stable_fingerprint():
    first = navigation_fingerprint()
    assert len(first) == 64
    assert navigation_fingerprint() == first


def test_normal_sim_and_real_workspace_are_not_imported_by_shell_wrapper():
    root = Path(__file__).resolve().parents[3]
    source = (root / 'scripts/experiment.sh').read_text()
    assert 'ROS_DOMAIN_ID=113' in source
    assert 'terrain_real_ws' not in source
    assert 'exec env -i' in source


def test_nav2_standard_defaults_remain_original():
    root = Path(__file__).resolve().parents[3]
    source = (
        root /
        'src/launch_pkg/launch/terrain_navigation_nav2.launch.py').read_text()
    assert ("DeclareLaunchArgument('navigation_use_sim_time', "
            "default_value='false')") in source
    assert "default_value=default_common_parameters" in source
    assert "default_value=default_nav2_parameters" in source
    assert "'params.yaml'" in source and "'nav2_outdoor_params.yaml'" in source
