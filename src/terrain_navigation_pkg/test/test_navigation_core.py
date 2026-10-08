import math

import pytest

from terrain_navigation_pkg.navigation_core import ArrivalDebouncer
from terrain_navigation_pkg.navigation_core import EnuPoint
from terrain_navigation_pkg.navigation_core import GeodeticPoint
from terrain_navigation_pkg.navigation_core import bearing_from_north_deg
from terrain_navigation_pkg.navigation_core import compute_go_to_goal_command
from terrain_navigation_pkg.navigation_core import geodetic_to_enu
from terrain_navigation_pkg.navigation_core import horizontal_distance
from terrain_navigation_pkg.navigation_core import interpolate_line
from terrain_navigation_pkg.navigation_core import median_geodetic_point


def test_geodetic_to_enu_small_east_and_north_offsets():
    origin = GeodeticPoint(37.0, 127.0, 50.0)
    north = geodetic_to_enu(
        GeodeticPoint(37.0001, 127.0, 50.0), origin
    )
    east = geodetic_to_enu(
        GeodeticPoint(37.0, 127.0001, 50.0), origin
    )

    assert north.north_m == pytest.approx(11.10, abs=0.05)
    assert abs(north.east_m) < 0.01
    assert east.east_m == pytest.approx(8.90, abs=0.05)
    assert abs(east.north_m) < 0.01


def test_carla_mercator_matches_simulator_equatorial_scale():
    origin = GeodeticPoint(0.0, 0.0, 10.0)
    offset = geodetic_to_enu(
        GeodeticPoint(0.0001, 0.0001, 12.5),
        origin,
        'carla_mercator',
    )

    # CARLA's legacy Mercator georeference uses the WGS84 semi-major radius
    # in both planar axes at the equator (about 11.132 m per 0.0001 degree).
    assert offset.east_m == pytest.approx(11.13195, abs=0.0001)
    assert offset.north_m == pytest.approx(11.13195, abs=0.0001)
    assert offset.up_m == pytest.approx(2.5)


def test_wgs84_remains_default_and_differs_from_carla_north_scale():
    origin = GeodeticPoint(0.0, 0.0, 0.0)
    point = GeodeticPoint(0.0001, 0.0, 0.0)

    production = geodetic_to_enu(point, origin)
    simulator = geodetic_to_enu(point, origin, 'carla_mercator')

    assert production.north_m == pytest.approx(11.05743, abs=0.0001)
    assert simulator.north_m - production.north_m > 0.07


def test_geodetic_to_enu_rejects_unknown_projection():
    with pytest.raises(ValueError, match='projection_mode'):
        geodetic_to_enu(
            GeodeticPoint(37.0, 127.0),
            GeodeticPoint(37.0, 127.0),
            'automatic',
        )


def test_median_geodetic_point_rejects_single_outlier():
    points = [
        GeodeticPoint(37.000000, 127.000000, 10.0),
        GeodeticPoint(37.000001, 127.000002, 10.2),
        GeodeticPoint(36.999999, 126.999998, 9.8),
        GeodeticPoint(38.000000, 128.000000, 1000.0),
        GeodeticPoint(37.000002, 127.000001, 10.1),
    ]

    origin = median_geodetic_point(points)

    assert origin.latitude_deg == pytest.approx(37.000001)
    assert origin.longitude_deg == pytest.approx(127.000001)
    assert origin.altitude_m == pytest.approx(10.1)


def test_median_geodetic_point_requires_samples():
    with pytest.raises(ValueError):
        median_geodetic_point([])


def test_bearing_uses_north_clockwise_convention():
    origin = EnuPoint(0.0, 0.0)
    assert bearing_from_north_deg(origin, EnuPoint(0.0, 1.0)) == 0.0
    assert bearing_from_north_deg(origin, EnuPoint(1.0, 0.0)) == 90.0
    assert bearing_from_north_deg(origin, EnuPoint(0.0, -1.0)) == 180.0
    assert bearing_from_north_deg(origin, EnuPoint(-1.0, 0.0)) == 270.0


def test_interpolate_line_includes_both_endpoints():
    points = interpolate_line(
        EnuPoint(0.0, 0.0), EnuPoint(0.0, 2.5), 1.0
    )
    assert len(points) == 4
    assert points[0] == EnuPoint(0.0, 0.0, 0.0)
    assert points[-1] == EnuPoint(0.0, 2.5, 0.0)
    assert horizontal_distance(points[0], points[-1]) == 2.5


def test_arrival_debouncer_requires_repeated_samples_and_hysteresis():
    tracker = ArrivalDebouncer(3.0, 4.5, 3)
    assert not tracker.update(2.9)
    assert not tracker.update(2.8)
    assert tracker.update(2.7)
    assert tracker.update(4.0)
    assert not tracker.update(4.6)
    assert not tracker.update(2.0)


def test_arrival_debouncer_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        ArrivalDebouncer(3.0, 2.0, 1)
    with pytest.raises(ValueError):
        ArrivalDebouncer(3.0, 4.0, 0)
    with pytest.raises(ValueError):
        ArrivalDebouncer(3.0, 4.0, 1).update(math.nan)


def _command(east, north, yaw=0.0, distance=20.0):
    return compute_go_to_goal_command(
        east,
        north,
        yaw,
        distance,
        max_speed_mps=1.5,
        min_speed_mps=0.35,
        slowdown_distance_m=8.0,
        stop_distance_m=3.0,
        heading_kp=1.0,
        max_yaw_rate_rps=0.5,
        minimum_heading_speed_ratio=0.2,
        wheelbase_m=2.85,
        maximum_steering_angle_rad=math.radians(35.0),
        recovery_heading_threshold_rad=math.radians(75.0),
        recovery_minimum_speed_mps=0.6,
    )


def test_go_to_goal_drives_straight_toward_east():
    command = _command(20.0, 0.0)
    assert command.speed_mps == pytest.approx(1.5)
    assert command.yaw_rate_rps == pytest.approx(0.0)


def test_go_to_goal_uses_positive_ros_yaw_rate_for_left_turn():
    command = _command(0.0, 20.0)
    assert command.speed_mps == pytest.approx(0.6)
    assert command.yaw_rate_rps == pytest.approx(
        0.6 * math.tan(math.radians(35.0)) / 2.85
    )


def test_go_to_goal_recovery_respects_ackermann_turning_radius():
    command = _command(-20.0, 0.0)
    curvature = abs(command.yaw_rate_rps) / command.speed_mps
    assert command.speed_mps >= 0.6
    assert curvature == pytest.approx(
        math.tan(math.radians(35.0)) / 2.85
    )


def test_go_to_goal_stops_inside_arrival_radius():
    command = _command(2.0, 0.0, distance=2.0)
    assert command.speed_mps == 0.0
    assert command.yaw_rate_rps == 0.0


def test_go_to_goal_wraps_heading_error_across_pi():
    command = _command(-10.0, -0.1, yaw=math.pi - 0.01)
    assert abs(command.heading_error_rad) < 0.03


def test_go_to_goal_caps_curvature_during_final_approach():
    command = compute_go_to_goal_command(
        3.0,
        4.0,
        0.0,
        5.0,
        max_speed_mps=1.5,
        min_speed_mps=0.35,
        slowdown_distance_m=8.0,
        stop_distance_m=3.0,
        heading_kp=1.0,
        max_yaw_rate_rps=0.5,
        minimum_heading_speed_ratio=0.2,
        arrival_maximum_curvature_per_m=0.10,
    )
    assert abs(command.yaw_rate_rps) / command.speed_mps <= 0.10 + 1e-9
