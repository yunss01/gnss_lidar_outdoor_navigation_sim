"""Pure navigation math and state helpers.

The ROS node keeps these functions separate so the coordinate conversion and
arrival logic can be tested without a running ROS graph.
"""

from dataclasses import dataclass
import math
from statistics import median
from typing import Iterable, List, Tuple


WGS84_SEMI_MAJOR_M = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_ECCENTRICITY_SQUARED = (
    WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
)
MEAN_EARTH_RADIUS_M = 6371008.8
GEODETIC_PROJECTION_MODES = ('wgs84', 'carla_mercator')


@dataclass(frozen=True)
class GeodeticPoint:
    """A WGS84 latitude/longitude/altitude point."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0


@dataclass(frozen=True)
class EnuPoint:
    """A local east/north/up coordinate in metres."""

    east_m: float
    north_m: float
    up_m: float = 0.0


@dataclass(frozen=True)
class GoToGoalCommand:
    """A forward speed and ROS-positive-left yaw-rate command."""

    speed_mps: float
    yaw_rate_rps: float
    heading_error_rad: float


def median_geodetic_point(points: Iterable[GeodeticPoint]) -> GeodeticPoint:
    """Return a component-wise median GNSS point.

    A median is deliberately used for the stationary startup anchor because
    one noisy GNSS fix must not translate every stored waypoint in the local
    navigation frame. It also rejects isolated position or altitude outliers
    without requiring sensor-specific covariance tuning.
    """
    point_list = list(points)
    if not point_list:
        raise ValueError('at least one GNSS point is required')
    for point in point_list:
        validate_geodetic(point)
    return GeodeticPoint(
        median(point.latitude_deg for point in point_list),
        median(point.longitude_deg for point in point_list),
        median(point.altitude_m for point in point_list),
    )


def normalize_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    if not math.isfinite(angle_rad):
        raise ValueError('angle_rad must be finite')
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def compute_go_to_goal_command(
    goal_east_m: float,
    goal_north_m: float,
    current_yaw_rad: float,
    distance_m: float,
    max_speed_mps: float,
    min_speed_mps: float,
    slowdown_distance_m: float,
    stop_distance_m: float,
    heading_kp: float,
    max_yaw_rate_rps: float,
    minimum_heading_speed_ratio: float,
    wheelbase_m: float = 2.85,
    maximum_steering_angle_rad: float = math.radians(35.0),
    recovery_heading_threshold_rad: float = math.radians(75.0),
    recovery_minimum_speed_mps: float = 0.6,
    arrival_maximum_curvature_per_m=None,
) -> GoToGoalCommand:
    """Compute a conservative Ackermann-compatible goal command.

    ``goal_east_m`` and ``goal_north_m`` are the components of the local ENU
    vector from the vehicle to the goal. The vehicle always drives forward;
    large heading errors therefore produce a slow steering arc instead of a
    rotate-in-place command that a car cannot execute.
    """
    values = (
        goal_east_m,
        goal_north_m,
        current_yaw_rad,
        distance_m,
        max_speed_mps,
        min_speed_mps,
        slowdown_distance_m,
        stop_distance_m,
        heading_kp,
        max_yaw_rate_rps,
        minimum_heading_speed_ratio,
        wheelbase_m,
        maximum_steering_angle_rad,
        recovery_heading_threshold_rad,
        recovery_minimum_speed_mps,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('go-to-goal inputs must be finite')
    if distance_m < 0.0:
        raise ValueError('distance_m must be non-negative')
    if not 0.0 <= min_speed_mps <= max_speed_mps:
        raise ValueError('speed limits must satisfy 0 <= min <= max')
    if stop_distance_m < 0.0:
        raise ValueError('stop_distance_m must be non-negative')
    if slowdown_distance_m <= stop_distance_m:
        raise ValueError('slowdown distance must exceed stop distance')
    if heading_kp < 0.0 or max_yaw_rate_rps < 0.0:
        raise ValueError('heading gains and limits must be non-negative')
    if not 0.0 <= minimum_heading_speed_ratio <= 1.0:
        raise ValueError('minimum heading speed ratio must be in [0, 1]')
    if wheelbase_m <= 0.0:
        raise ValueError('wheelbase_m must be positive')
    if not 0.0 < maximum_steering_angle_rad < 0.5 * math.pi:
        raise ValueError('maximum steering angle must be in (0, pi/2)')
    if not 0.0 < recovery_heading_threshold_rad <= math.pi:
        raise ValueError('recovery heading threshold must be in (0, pi]')
    if not 0.0 <= recovery_minimum_speed_mps <= max_speed_mps:
        raise ValueError('recovery minimum speed must be within speed limits')
    if (
        arrival_maximum_curvature_per_m is not None
        and (
            not math.isfinite(arrival_maximum_curvature_per_m)
            or arrival_maximum_curvature_per_m <= 0.0
        )
    ):
        raise ValueError('arrival maximum curvature must be positive')

    desired_yaw = math.atan2(goal_north_m, goal_east_m)
    heading_error = normalize_angle_rad(desired_yaw - current_yaw_rad)
    if distance_m <= stop_distance_m:
        return GoToGoalCommand(0.0, 0.0, heading_error)

    distance_ratio = min(
        1.0,
        max(
            0.0,
            (distance_m - stop_distance_m)
            / (slowdown_distance_m - stop_distance_m),
        ),
    )
    distance_speed = (
        min_speed_mps
        + (max_speed_mps - min_speed_mps) * distance_ratio
    )
    heading_ratio = max(
        minimum_heading_speed_ratio,
        max(0.0, math.cos(abs(heading_error))),
    )
    speed = distance_speed * heading_ratio
    if abs(heading_error) >= recovery_heading_threshold_rad:
        # A car cannot rotate in place. Keep enough forward motion for a
        # visible U-turn when the destination lies behind the vehicle.
        speed = max(speed, recovery_minimum_speed_mps)
    maximum_curvature = math.tan(maximum_steering_angle_rad) / wheelbase_m
    if (
        arrival_maximum_curvature_per_m is not None
        and distance_m < slowdown_distance_m
    ):
        maximum_curvature = min(
            maximum_curvature,
            arrival_maximum_curvature_per_m,
        )
    ackermann_yaw_rate_limit = speed * maximum_curvature
    yaw_rate_limit = min(max_yaw_rate_rps, ackermann_yaw_rate_limit)
    yaw_rate = max(
        -yaw_rate_limit,
        min(yaw_rate_limit, heading_kp * heading_error),
    )
    return GoToGoalCommand(speed, yaw_rate, heading_error)


def validate_geodetic(point: GeodeticPoint) -> None:
    """Raise ``ValueError`` when a geodetic coordinate is invalid."""
    values = (
        point.latitude_deg,
        point.longitude_deg,
        point.altitude_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('GNSS coordinates must be finite')
    if not -90.0 <= point.latitude_deg <= 90.0:
        raise ValueError('latitude must be in [-90, 90] degrees')
    if not -180.0 <= point.longitude_deg <= 180.0:
        raise ValueError('longitude must be in [-180, 180] degrees')


def geodetic_to_ecef(point: GeodeticPoint) -> Tuple[float, float, float]:
    """Convert a WGS84 coordinate to Earth-centred Earth-fixed metres."""
    validate_geodetic(point)
    latitude = math.radians(point.latitude_deg)
    longitude = math.radians(point.longitude_deg)
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    prime_vertical_radius = WGS84_SEMI_MAJOR_M / math.sqrt(
        1.0 - WGS84_ECCENTRICITY_SQUARED * sin_latitude ** 2
    )
    x_value = (
        prime_vertical_radius + point.altitude_m
    ) * cos_latitude * math.cos(longitude)
    y_value = (
        prime_vertical_radius + point.altitude_m
    ) * cos_latitude * math.sin(longitude)
    z_value = (
        prime_vertical_radius
        * (1.0 - WGS84_ECCENTRICITY_SQUARED)
        + point.altitude_m
    ) * sin_latitude
    return x_value, y_value, z_value


def geodetic_to_enu(
    point: GeodeticPoint,
    origin: GeodeticPoint,
    projection_mode: str = 'wgs84',
) -> EnuPoint:
    """Convert a geodetic point to a local east/north/up frame.

    ``wgs84`` is the production conversion for real receivers. CARLA's GNSS
    actor uses its legacy Mercator georeference scale; near the equator this
    differs from the WGS84 meridional scale by about 0.7 percent. The explicit
    ``carla_mercator`` mode reproduces that simulator contract without
    weakening or silently changing the real-vehicle default.
    """
    validate_geodetic(point)
    validate_geodetic(origin)
    if projection_mode not in GEODETIC_PROJECTION_MODES:
        raise ValueError(
            'projection_mode must be one of: {}'.format(
                ', '.join(GEODETIC_PROJECTION_MODES)
            )
        )
    if projection_mode == 'carla_mercator':
        point_latitude = math.radians(point.latitude_deg)
        origin_latitude = math.radians(origin.latitude_deg)
        if (
            abs(point_latitude) >= 0.5 * math.pi
            or abs(origin_latitude) >= 0.5 * math.pi
        ):
            raise ValueError(
                'carla_mercator is undefined at the geographic poles'
            )

        # This follows CARLA's legacy LatLonToMercator convention: scale the
        # Web-Mercator plane by cos(reference latitude). It is deliberately a
        # simulator-only projection, not an approximation used by hardware.
        scale = math.cos(origin_latitude)
        longitude_delta = math.radians(
            point.longitude_deg - origin.longitude_deg
        )

        def mercator_y(latitude_rad):
            return math.asinh(math.tan(latitude_rad))

        return EnuPoint(
            WGS84_SEMI_MAJOR_M * scale * longitude_delta,
            WGS84_SEMI_MAJOR_M * scale * (
                mercator_y(point_latitude)
                - mercator_y(origin_latitude)
            ),
            point.altitude_m - origin.altitude_m,
        )

    point_ecef = geodetic_to_ecef(point)
    origin_ecef = geodetic_to_ecef(origin)
    delta_x = point_ecef[0] - origin_ecef[0]
    delta_y = point_ecef[1] - origin_ecef[1]
    delta_z = point_ecef[2] - origin_ecef[2]

    latitude = math.radians(origin.latitude_deg)
    longitude = math.radians(origin.longitude_deg)
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    sin_longitude = math.sin(longitude)
    cos_longitude = math.cos(longitude)

    east = -sin_longitude * delta_x + cos_longitude * delta_y
    north = (
        -sin_latitude * cos_longitude * delta_x
        - sin_latitude * sin_longitude * delta_y
        + cos_latitude * delta_z
    )
    up = (
        cos_latitude * cos_longitude * delta_x
        + cos_latitude * sin_longitude * delta_y
        + sin_latitude * delta_z
    )
    return EnuPoint(east, north, up)


def horizontal_distance(first: EnuPoint, second: EnuPoint) -> float:
    """Return planar ENU distance in metres."""
    return math.hypot(
        second.east_m - first.east_m,
        second.north_m - first.north_m,
    )


def bearing_from_north_deg(first: EnuPoint, second: EnuPoint) -> float:
    """Return clockwise bearing where north is 0 degrees."""
    east_delta = second.east_m - first.east_m
    north_delta = second.north_m - first.north_m
    if east_delta == 0.0 and north_delta == 0.0:
        return 0.0
    return math.degrees(math.atan2(east_delta, north_delta)) % 360.0


def yaw_from_east_rad(first: EnuPoint, second: EnuPoint) -> float:
    """Return ROS ENU yaw where east (+x) is zero radians."""
    return math.atan2(
        second.north_m - first.north_m,
        second.east_m - first.east_m,
    )


def interpolate_line(
    first: EnuPoint,
    second: EnuPoint,
    spacing_m: float,
) -> List[EnuPoint]:
    """Sample the straight line between two ENU points."""
    if spacing_m <= 0.0:
        raise ValueError('spacing_m must be positive')
    distance = horizontal_distance(first, second)
    segment_count = max(1, int(math.ceil(distance / spacing_m)))
    return [
        EnuPoint(
            first.east_m
            + (second.east_m - first.east_m) * index / segment_count,
            first.north_m
            + (second.north_m - first.north_m) * index / segment_count,
            first.up_m
            + (second.up_m - first.up_m) * index / segment_count,
        )
        for index in range(segment_count + 1)
    ]


class ArrivalDebouncer:
    """Add hysteresis and repeated observations to noisy GNSS arrival."""

    def __init__(
        self,
        arrival_radius_m: float,
        leave_radius_m: float,
        required_consecutive_samples: int,
    ):
        if arrival_radius_m <= 0.0:
            raise ValueError('arrival_radius_m must be positive')
        if leave_radius_m <= arrival_radius_m:
            raise ValueError(
                'leave_radius_m must be greater than arrival_radius_m'
            )
        if required_consecutive_samples < 1:
            raise ValueError(
                'required_consecutive_samples must be at least one'
            )
        self.arrival_radius_m = float(arrival_radius_m)
        self.leave_radius_m = float(leave_radius_m)
        self.required_consecutive_samples = int(
            required_consecutive_samples
        )
        self.reset()

    def reset(self) -> None:
        """Reset the reached state for a new goal."""
        self.inside_count = 0
        self.reached = False

    def update(self, distance_m: float) -> bool:
        """Update and return the debounced arrival state."""
        if not math.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError('distance_m must be finite and non-negative')
        if self.reached:
            if distance_m > self.leave_radius_m:
                self.reached = False
                self.inside_count = 0
            return self.reached

        if distance_m <= self.arrival_radius_m:
            self.inside_count += 1
            if self.inside_count >= self.required_consecutive_samples:
                self.reached = True
        else:
            self.inside_count = 0
        return self.reached


def finite_mean(values: Iterable[float]) -> float:
    """Return a mean while rejecting invalid input."""
    materialized = [float(value) for value in values]
    if not materialized or not all(
        math.isfinite(value) for value in materialized
    ):
        raise ValueError('values must be a non-empty finite sequence')
    return sum(materialized) / len(materialized)
