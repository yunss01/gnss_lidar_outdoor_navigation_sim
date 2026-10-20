"""Capture a fixed straight route and manage ONLY this experiment's actors."""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from .controlled_carla_obstacle import (
    _ground_hit,
    _select_vehicle,
    import_carla,
)
from .simulation_experiment_core import (
    rectangle,
    scenario_offsets,
    validate_anchor,
    wrap_angle,
)


def write_json(path, value, exclusive=False):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value,
                      stream,
                      indent=2,
                      ensure_ascii=False,
                      allow_nan=False)
            stream.write('\n')
        return
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) +
        '\n',
        encoding='utf-8')
    temporary.replace(path)


def read_anchor(path):
    return validate_anchor(json.loads(Path(path).expanduser().read_text()))


def connect(options):
    carla = import_carla(Path(options.carla_root).expanduser())
    client = carla.Client(options.host, options.port)
    client.set_timeout(10.0)
    world = client.get_world()
    # A new client's actor list is empty until its first world snapshot.
    # Observe the existing client's tick; never advance the world ourselves.
    try:
        world.wait_for_tick(3.0)
    except RuntimeError as error:
        raise RuntimeError(
            f'No CARLA frame received from {options.host}:{options.port} '
            'within 3 seconds. Keep terminal B experiment_manual.py running '
            'before capture/prepare/run.') from error
    return carla, world


def state_path(anchor_path):
    return Path(anchor_path).expanduser().parent / 'prepared.json'


def fingerprint(anchor):
    return hashlib.sha256(json.dumps(anchor,
                                     sort_keys=True).encode()).hexdigest()


def _speed(actor):
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def _transform_dict(transform):
    return {
        key: float(getattr(transform.location, key))
        for key in ('x', 'y', 'z')
    } | {
        key: float(getattr(transform.rotation, key))
        for key in ('yaw', 'pitch', 'roll')
    }


def _wait_for_placement(world, actor, expected, timeout_s=3.0):
    """Observe a fresh snapshot containing the final frozen-actor pose."""
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            snapshot = world.wait_for_tick(min(1.0, remaining))
        except RuntimeError:
            continue
        if snapshot.find(actor.id) is None:
            continue
        current = actor.get_transform()
        distance = math.sqrt(sum(
            (getattr(current.location, key) -
             getattr(expected.location, key))**2 for key in ('x', 'y', 'z')))
        heading = abs(wrap_angle(math.radians(
            current.rotation.yaw - expected.rotation.yaw)))
        if distance <= 0.005 and heading <= math.radians(0.1):
            return current
    raise RuntimeError(
        f'obstacle {actor.id} placement was not confirmed in a fresh '
        'CARLA frame; keep terminal B running')


def box_record(actor, transform=None):
    box = actor.bounding_box
    if transform is None:
        transform = actor.get_transform()
    center = transform.transform(box.location)
    yaw = -math.radians(transform.rotation.yaw + box.rotation.yaw)
    return {
        'actor_id': actor.id,
        'blueprint': actor.type_id,
        'role_name': actor.attributes.get('role_name', ''),
        'polygon_ros': rectangle(center.x, -center.y, yaw, box.extent.x,
                                 box.extent.y),
        'transform_carla': _transform_dict(transform),
        'half_length_m': float(box.extent.x),
        'half_width_m': float(box.extent.y),
    }


def verify_hero(world, anchor):
    if world.get_map().name != anchor['map_name']:
        raise RuntimeError(
            'map differs from anchor; do not silently change worlds')
    hero = _select_vehicle(world)
    if hero.type_id != anchor['vehicle_blueprint']:
        raise RuntimeError('hero blueprint differs from anchor')
    pose = hero.get_transform()
    start = anchor['start_carla']
    if math.hypot(pose.location.x - start['x'],
                  pose.location.y - start['y']) > 0.25:
        raise RuntimeError(
            'hero not at anchor; restart experiment_manual.py --anchor FILE')
    if abs(wrap_angle(math.radians(pose.rotation.yaw -
                                   start['yaw']))) > math.radians(0.5):
        raise RuntimeError('hero heading differs from anchor')
    if _speed(hero) > 0.05:
        raise RuntimeError('hero must be stationary')
    return hero


def gnss_measurement_origin(world, hero):
    """Match the existing GNSS-origin anchor, not a body/antenna mix-up."""
    sensors = [
        actor for actor in world.get_actors().filter('sensor.other.gnss')
        if actor.parent is not None and actor.parent.id == hero.id
    ]
    if len(sensors) != 1:
        raise RuntimeError('need exactly one GNSS sensor attached to the hero')
    return sensors[0].get_transform().location


def capture(options):
    carla, world = connect(options)
    hero = _select_vehicle(world)
    if _speed(hero) > 0.05:
        raise RuntimeError('park the hero before capturing an anchor')
    transform = hero.get_transform()
    start = _transform_dict(transform)
    yaw = math.radians(start['yaw'])
    gnss_origin = gnss_measurement_origin(world, hero)
    waypoints = []
    lane_samples = []
    for distance in (0.0, options.route_length_m):
        point = carla.Location(gnss_origin.x + math.cos(yaw) * distance,
                               gnss_origin.y + math.sin(yaw) * distance,
                               gnss_origin.z)
        geo = world.get_map().transform_to_geolocation(point)
        waypoints.append({
            'latitude': geo.latitude,
            'longitude': geo.longitude,
            'altitude': geo.altitude
        })
    for distance in range(0, int(options.route_length_m) + 1, 2):
        point = carla.Location(start['x'] + math.cos(yaw) * distance,
                               start['y'] + math.sin(yaw) * distance,
                               start['z'])
        waypoint = world.get_map().get_waypoint(
            point, project_to_road=False, lane_type=carla.LaneType.Driving)
        lane_samples.append(waypoint is not None)
    if not all(lane_samples):
        raise RuntimeError(
            'the proposed straight A-B leaves a driving lane; '
            'choose a straight, clear >=40 m road and capture again')
    box = hero.bounding_box
    anchor = {
        'schema_version': 1,
        'created_at': datetime.now().astimezone().isoformat(),
        'map_name': world.get_map().name,
        'vehicle_blueprint': hero.type_id,
        'start_carla': start,
        'start_ros': {
            'x': start['x'],
            'y': -start['y'],
            'yaw_rad': -yaw
        },
        'gnss_origin_carla': dict(x=gnss_origin.x,
                                  y=gnss_origin.y,
                                  z=gnss_origin.z),
        'route_coordinate_contract': (
            'A/B GNSS antenna positions for a fixed body displacement; '
            'truth evaluator uses body A/B, not antenna coordinates'),
        'route_length_m': options.route_length_m,
        'obstacle_distance_m': options.obstacle_distance_m,
        'corridor_half_width_m': options.corridor_half_width_m,
        'waypoints': waypoints,
        'driving_lane_check': True,
        'ego_box': {
            'half_length_m': float(box.extent.x),
            'half_width_m': float(box.extent.y),
            'offset_forward_m': float(box.location.x),
            'offset_left_m': -float(box.location.y),
            'yaw_offset_rad': -math.radians(box.rotation.yaw)
        },
        'physics_step_s': 0.05,
        'policy': ('Lincoln sensitivity test; '
                   'not calibrated real-vehicle dynamics'),
    }
    validate_anchor(anchor)
    write_json(options.anchor, anchor, exclusive=True)
    print(f'Captured fixed A-B anchor: {Path(options.anchor).expanduser()}')
    print('Centreline lane check passed. '
          'Verify clear road and lateral room in Pygame.')


def straight_start_candidates(carla, world):
    """Read-only search; lane membership is NOT a clearance proof."""
    found = []
    for transform in world.get_map().get_spawn_points():
        yaw = math.radians(transform.rotation.yaw)
        good = True
        for distance in range(0, 41, 2):
            waypoint = world.get_map().get_waypoint(
                carla.Location(transform.location.x + math.cos(yaw) * distance,
                               transform.location.y + math.sin(yaw) * distance,
                               transform.location.z),
                project_to_road=False,
                lane_type=carla.LaneType.Driving)
            if waypoint is None or waypoint.is_junction:
                good = False
                break
        if good:
            found.append(transform)
            if len(found) >= 5:
                break
    if not found:
        raise RuntimeError(
            'no 40 m non-junction lane candidate found in current map')
    return found


def find_start(options):
    carla, world = connect(options)
    for transform in straight_start_candidates(carla, world):
        print('--spawn-point=' + ','.join(
            f'{value:.3f}'
            for value in (transform.location.x, transform.location.y,
                          transform.location.z, transform.rotation.yaw)))
    print('Candidates only: inspect lateral room and other objects '
          'before capture/run.')


def prepare(options):
    anchor = read_anchor(options.anchor)
    carla, world = connect(options)
    hero = verify_hero(world, anchor)
    destination = state_path(options.anchor)
    if destination.exists():
        old = json.loads(destination.read_text())
        if any(
                world.get_actor(int(record['actor_id'])) is not None
                for record in old.get('obstacles', [])):
            raise RuntimeError(
                'previous obstacles still exist; use cleanup first')
    box = hero.bounding_box
    offsets = scenario_offsets(options.scenario, anchor['obstacle_distance_m'],
                               2 * box.extent.y,
                               anchor['corridor_half_width_m'])
    role = 'terrain_experiment_' + uuid.uuid4().hex[:12]
    state = {
        'schema_version': 1,
        'scenario': options.scenario,
        'anchor_sha256': fingerprint(anchor),
        'map_name': anchor['map_name'],
        'hero_actor_id': hero.id,
        'role_name': role,
        'prepared': False,
        'obstacles': [],
        'created_at': datetime.now().astimezone().isoformat()
    }
    write_json(destination, state)
    start = anchor['start_carla']
    yaw = math.radians(start['yaw'])
    created = []
    try:
        for distance, right in offsets:
            x = start['x'] + distance * math.cos(yaw) - right * math.sin(yaw)
            y = start['y'] + distance * math.sin(yaw) + right * math.cos(yaw)
            surface = _ground_hit(world, carla, x, y, start['z'])
            if str(surface.label) not in {
                    'Roads', 'Ground', 'Terrain', 'RoadLines', 'Sidewalks'
            }:
                raise RuntimeError(
                    f'placement is not a road/ground surface: {surface.label}')
            blueprint = world.get_blueprint_library().find(hero.type_id)
            blueprint.set_attribute('role_name', role)
            actor = world.try_spawn_actor(
                blueprint,
                carla.Transform(carla.Location(x, y, surface.location.z + 2),
                                carla.Rotation(yaw=start['yaw'])))
            if actor is None:
                raise RuntimeError(
                    'obstacle spawn rejected; nothing else will be removed')
            created.append(actor)
            actor.set_autopilot(False)
            actor.set_simulate_physics(False)
            lowest = (actor.bounding_box.location.z -
                      actor.bounding_box.extent.z)
            placement = carla.Transform(
                carla.Location(x, y, surface.location.z - lowest + 0.02),
                carla.Rotation(yaw=start['yaw']))
            actor.set_transform(placement)
            actor.apply_control(
                carla.VehicleControl(brake=1.0, hand_brake=True))
            confirmed = _wait_for_placement(world, actor, placement)
            state['obstacles'].append(box_record(actor, confirmed))
            write_json(destination, state)
        state['prepared'] = True
        state['corridor_contract'] = (
            'S3 closes the allowed +/- corridor, not the whole CARLA world; '
            'an out-of-corridor detour fails the evaluator')
        write_json(destination, state)
        archive = destination.parent / 'preparations' / f'{role}.json'
        write_json(archive, state, exclusive=True)
    except Exception:
        for actor in created:
            actor.destroy()
        state['prepared'] = False
        state['error'] = 'preparation failed; own spawned actors removed'
        write_json(destination, state)
        raise
    print(f'Prepared {options.scenario}: {len(offsets)} frozen test vehicles')
    print(f'State: {destination}')
    print('Check obstacle placement/point cloud before starting the route.')


def validate_prepared(world, anchor, anchor_path, scenario):
    hero = verify_hero(world, anchor)
    state = json.loads(state_path(anchor_path).read_text())
    if (not state.get('prepared') or state.get('scenario') != scenario
            or state.get('anchor_sha256') != fingerprint(anchor)
            or state.get('map_name') != world.get_map().name
            or state.get('hero_actor_id') != hero.id):
        raise RuntimeError(
            'prepared scenario does not match anchor/current hero')
    for record in state['obstacles']:
        actor = world.get_actor(record['actor_id'])
        if (actor is None or actor.type_id != record['blueprint']
                or actor.attributes.get('role_name') != state['role_name']):
            raise RuntimeError('prepared actor missing or identity changed')
        current = actor.get_transform()
        expected = record['transform_carla']
        distance = math.sqrt(sum(
            (getattr(current.location, key) - expected[key])**2
            for key in ('x', 'y', 'z')))
        heading = abs(wrap_angle(math.radians(
            current.rotation.yaw - expected['yaw'])))
        if distance > 0.05 or heading > math.radians(0.5):
            raise RuntimeError(
                f'prepared obstacle moved: actor={actor.id}, '
                f'position_delta={distance:.3f}m, '
                f'yaw_delta={math.degrees(heading):.3f}deg')
    return state


def cleanup(options):
    _carla, world = connect(options)
    path = state_path(options.anchor)
    if not path.exists():
        print('No prepared state; no actors changed.')
        return
    state = json.loads(path.read_text())
    if state['map_name'] != world.get_map().name:
        raise RuntimeError(
            'different world; refusing potentially reused actor IDs')
    targets = []
    for record in state['obstacles']:
        actor = world.get_actor(int(record['actor_id']))
        if actor is None:
            continue
        if (actor.type_id != record['blueprint']
                or actor.attributes.get('role_name') != state['role_name']):
            raise RuntimeError(
                'identity mismatch; no unrelated actor will be destroyed')
        targets.append(actor)
    hero = world.get_actor(state['hero_actor_id'])
    if hero is not None and _speed(hero) > 0.05:
        raise RuntimeError('stop the hero before removing obstacles')
    for actor in targets:
        if not actor.destroy():
            raise RuntimeError(
                f'CARLA refused to remove owned actor {actor.id}')
    state['prepared'] = False
    state['cleaned_at'] = datetime.now().astimezone().isoformat()
    write_json(path, state)
    print(f'Removed {len(targets)} experiment-owned vehicles. '
          'Scenario records retained; prepare recreates them.')
