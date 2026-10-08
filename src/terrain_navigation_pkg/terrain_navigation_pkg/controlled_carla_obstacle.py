"""Place one reproducible CARLA prop relative to the operator vehicle."""

import argparse
import importlib
import json
import math
from pathlib import Path
import sys


DEFAULT_CARLA_ROOT = Path('/home/sukja/carla/0.10.0')
DEFAULT_STATE_FILE = Path(
    '/home/sukja/terrain_nav_data/learning/controlled_obstacle/'
    'active_actor.json'
)
GROUND_LABELS = {
    'Ground', 'RoadLines', 'Roads', 'Sidewalks', 'Terrain',
}


def planar_target(origin_x, origin_y, yaw_degrees, forward_m, right_m):
    """Return an XY target using CARLA's forward/right convention."""
    yaw_radians = math.radians(float(yaw_degrees))
    forward_x = math.cos(yaw_radians)
    forward_y = math.sin(yaw_radians)
    right_x = -forward_y
    right_y = forward_x
    return (
        float(origin_x) + float(forward_m) * forward_x
        + float(right_m) * right_x,
        float(origin_y) + float(forward_m) * forward_y
        + float(right_m) * right_y,
    )


def _carla_wheel(carla_root):
    python_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    wheels = sorted(
        (Path(carla_root) / 'PythonAPI' / 'carla' / 'dist').glob(
            f'carla-*-{python_tag}-{python_tag}-linux_x86_64.whl'
        )
    )
    if not wheels:
        raise RuntimeError(
            f'No CARLA wheel for {python_tag} under {carla_root}'
        )
    return wheels[-1]


def import_carla(carla_root):
    """Import CARLA, adding the matching local wheel when necessary."""
    try:
        return importlib.import_module('carla')
    except ImportError:
        sys.path.insert(0, str(_carla_wheel(carla_root)))
        return importlib.import_module('carla')


def _select_vehicle(world, requested_id=None):
    if requested_id is not None:
        actor = world.get_actor(int(requested_id))
        if actor is None or not actor.type_id.startswith('vehicle.'):
            raise RuntimeError(
                f'CARLA vehicle actor {requested_id} was not found'
            )
        return actor

    vehicles = list(world.get_actors().filter('vehicle.*'))
    heroes = [
        actor for actor in vehicles
        if actor.attributes.get('role_name') == 'hero'
    ]
    if len(heroes) == 1:
        return heroes[0]
    if len(vehicles) == 1:
        return vehicles[0]
    details = ', '.join(
        f'{actor.id}:{actor.type_id}:'
        f'{actor.attributes.get("role_name", "")}'
        for actor in vehicles
    )
    raise RuntimeError(
        'Could not select one operator vehicle. Pass --vehicle-id. '
        f'Candidates: {details or "none"}'
    )


def _ground_hit(world, carla, x, y, reference_z):
    start = carla.Location(float(x), float(y), float(reference_z) + 10.0)
    end = carla.Location(float(x), float(y), float(reference_z) - 20.0)
    hits = list(world.cast_ray(start, end))
    ground_hits = [
        hit for hit in hits if str(hit.label) in GROUND_LABELS
    ]
    candidates = ground_hits or hits
    if not candidates:
        raise RuntimeError('No surface was found below the requested target')
    return max(candidates, key=lambda hit: float(hit.location.z))


def _resolve_blueprint(world, blueprint_id):
    matches = list(world.get_blueprint_library().filter(blueprint_id))
    exact = [item for item in matches if item.id == blueprint_id]
    if len(exact) != 1:
        available = ', '.join(item.id for item in matches)
        raise RuntimeError(
            f'Blueprint {blueprint_id!r} was not found exactly. '
            f'Matches: {available or "none"}'
        )
    blueprint = exact[0]
    if blueprint.has_attribute('role_name'):
        blueprint.set_attribute(
            'role_name', 'terrain_nav_controlled_obstacle'
        )
    return blueprint


def _load_state(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def _target_context(world, carla, args):
    vehicle = _select_vehicle(world, args.vehicle_id)
    vehicle_transform = vehicle.get_transform()
    x, y = planar_target(
        vehicle_transform.location.x,
        vehicle_transform.location.y,
        vehicle_transform.rotation.yaw,
        args.forward_m,
        args.right_m,
    )
    hit = _ground_hit(
        world, carla, x, y, vehicle_transform.location.z
    )
    return vehicle, vehicle_transform, x, y, hit


def _print_target(vehicle, transform, x, y, hit, args):
    print(f'vehicle_id: {vehicle.id}')
    print(
        'vehicle_pose: '
        f'x={transform.location.x:.3f}, y={transform.location.y:.3f}, '
        f'z={transform.location.z:.3f}, yaw={transform.rotation.yaw:.3f}'
    )
    print(
        'requested_offset: '
        f'forward={args.forward_m:.3f} m, right={args.right_m:.3f} m'
    )
    print(
        f'target_surface: x={x:.3f}, y={y:.3f}, '
        f'z={hit.location.z:.3f}, label={hit.label}'
    )


def command_preview(world, carla, args):
    context = _target_context(world, carla, args)
    _print_target(*context, args)


def command_spawn(world, carla, args):
    state_path = Path(args.state_file).expanduser()
    existing = _load_state(state_path)
    if existing is not None:
        actor = world.get_actor(int(existing['actor_id']))
        if actor is not None:
            raise RuntimeError(
                f'Controlled actor {actor.id} is still active. '
                'Run the remove command first.'
            )

    vehicle, vehicle_transform, x, y, hit = _target_context(
        world, carla, args
    )
    _print_target(vehicle, vehicle_transform, x, y, hit, args)
    blueprint = _resolve_blueprint(world, args.blueprint)
    rotation = carla.Rotation(
        0.0,
        float(vehicle_transform.rotation.yaw) + float(args.yaw_offset_deg),
        0.0,
    )
    initial_transform = carla.Transform(
        carla.Location(float(x), float(y), float(hit.location.z) + 2.0),
        rotation,
    )
    actor = world.try_spawn_actor(blueprint, initial_transform)
    if actor is None:
        raise RuntimeError(
            'CARLA rejected the spawn. Confirm that the target is empty.'
        )

    try:
        box = actor.bounding_box
        lowest_local_z = float(box.location.z) - float(box.extent.z)
        final_z = (
            float(hit.location.z) - lowest_local_z
            + float(args.clearance_m)
        )
        final_transform = carla.Transform(
            carla.Location(float(x), float(y), final_z), rotation
        )
        actor.set_transform(final_transform)
        state = {
            'actor_id': int(actor.id),
            'blueprint': actor.type_id,
            'clearance_m': float(args.clearance_m),
            'forward_m': float(args.forward_m),
            'map_name': world.get_map().name,
            'right_m': float(args.right_m),
            'surface_label': str(hit.label),
            'surface_z_m': float(hit.location.z),
            'vehicle_id': int(vehicle.id),
            'world_transform': {
                'x': float(x),
                'y': float(y),
                'z': float(final_z),
                'yaw': float(rotation.yaw),
            },
        }
        _write_state(state_path, state)
    except Exception:
        actor.destroy()
        raise

    print(f'spawned_actor_id: {actor.id}')
    print(f'blueprint: {actor.type_id}')
    print(
        'bounding_box_extent: '
        f'x={box.extent.x:.3f}, y={box.extent.y:.3f}, '
        f'z={box.extent.z:.3f}'
    )
    print(f'state_file: {state_path}')


def command_status(world, _carla, args):
    state_path = Path(args.state_file).expanduser()
    state = _load_state(state_path)
    if state is None:
        print('No controlled obstacle state is recorded.')
        return
    actor = world.get_actor(int(state['actor_id']))
    state['actor_active'] = actor is not None
    if actor is not None:
        transform = actor.get_transform()
        state['current_transform'] = {
            'x': float(transform.location.x),
            'y': float(transform.location.y),
            'z': float(transform.location.z),
            'yaw': float(transform.rotation.yaw),
        }
    print(json.dumps(state, indent=2, sort_keys=True))


def command_remove(world, _carla, args):
    state_path = Path(args.state_file).expanduser()
    state = _load_state(state_path)
    if state is None:
        print('No controlled obstacle state is recorded.')
        return
    actor_id = int(state['actor_id'])
    actor = world.get_actor(actor_id)
    if actor is None:
        print(f'Actor {actor_id} is already absent from CARLA.')
    else:
        if not actor.type_id.startswith('static.prop.'):
            raise RuntimeError(
                f'Refusing to remove unexpected actor type {actor.type_id}'
            )
        if not actor.destroy():
            raise RuntimeError(f'CARLA failed to remove actor {actor_id}')
        print(f'Removed controlled actor {actor_id} ({actor.type_id}).')
    state_path.unlink(missing_ok=True)


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            'Preview, spawn, inspect, or remove one controlled CARLA prop '
            'relative to the operator vehicle.'
        )
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--timeout-s', type=float, default=5.0)
    parser.add_argument('--carla-root', default=str(DEFAULT_CARLA_ROOT))
    parser.add_argument('--state-file', default=str(DEFAULT_STATE_FILE))
    subparsers = parser.add_subparsers(dest='command', required=True)

    for name in ('preview', 'spawn'):
        subparser = subparsers.add_parser(name)
        subparser.add_argument('--vehicle-id', type=int)
        subparser.add_argument('--forward-m', type=float, default=6.0)
        subparser.add_argument('--right-m', type=float, default=0.0)
        if name == 'spawn':
            subparser.add_argument(
                '--blueprint', default='static.prop.creasedbox01'
            )
            subparser.add_argument('--yaw-offset-deg', type=float, default=0.0)
            subparser.add_argument('--clearance-m', type=float, default=0.02)
    subparsers.add_parser('status')
    subparsers.add_parser('remove')
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    carla = import_carla(Path(args.carla_root).expanduser())
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(float(args.timeout_s))
    world = client.get_world()
    # A newly connected secondary client starts with an empty frame-0 cache
    # while manual_control.py owns the synchronous tick.  Wait for that
    # primary client's next tick; never call world.tick() from this tool.
    world.wait_for_tick(float(args.timeout_s))
    commands = {
        'preview': command_preview,
        'spawn': command_spawn,
        'status': command_status,
        'remove': command_remove,
    }
    try:
        commands[args.command](world, carla, args)
    except RuntimeError as error:
        raise SystemExit(f'ERROR: {error}') from error


if __name__ == '__main__':
    main()
