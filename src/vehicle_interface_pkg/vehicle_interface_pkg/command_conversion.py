"""Convert ROS velocity commands into the educational vehicle protocol."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ArduinoCommand:
    """Discrete steering and signed rear-motor PWM command."""

    steering: int
    left_pwm: int
    right_pwm: int


def clamp(value, lower, upper):
    """Return ``value`` limited to the closed interval."""
    return max(lower, min(upper, value))


def velocity_to_arduino_command(
    linear_mps,
    angular_rps,
    *,
    wheelbase_m,
    maximum_steering_angle_rad,
    maximum_steering_step,
    maximum_speed_mps,
    maximum_pwm,
    steering_direction=1.0,
    motor_direction=1.0,
    steering_left_command=None,
    steering_center_command=0,
    steering_right_command=None,
):
    """Convert a bicycle-model ``Twist`` into the three Arduino integers.

    ``angular_rps`` is the ROS yaw-rate command, not a steering angle.  The
    bicycle relation ``yaw_rate = speed * tan(steering) / wheelbase`` is used
    before the physical wheel angle is quantized to the Arduino step range.
    Both rear motors initially receive the same PWM, matching the H-Mobility
    educational vehicle.  Per-wheel calibration can be added after bench
    measurements are available.

    The reference ``driving.ino`` uses ``-7`` for full physical left, ``0``
    for centre, and ``+7`` for full physical right.  Supplying the three
    ``steering_*_command`` values enables a piecewise mapping to that protocol
    and also permits later asymmetric calibration without changing this code.
    Leaving both endpoint commands as ``None`` retains the original symmetric
    ``steering_direction * step`` behaviour for backwards compatibility.
    """
    if wheelbase_m <= 0.0:
        raise ValueError('wheelbase_m must be positive')
    if maximum_steering_angle_rad <= 0.0:
        raise ValueError('maximum_steering_angle_rad must be positive')
    if maximum_steering_step <= 0:
        raise ValueError('maximum_steering_step must be positive')
    if maximum_speed_mps <= 0.0:
        raise ValueError('maximum_speed_mps must be positive')
    if maximum_pwm <= 0 or maximum_pwm > 255:
        raise ValueError('maximum_pwm must be in 1..255')
    if not math.isfinite(float(steering_direction)):
        raise ValueError('steering_direction must be finite')
    if float(steering_direction) == 0.0:
        raise ValueError('steering_direction must be non-zero')

    explicit_steering_commands = (
        steering_left_command is not None
        or steering_right_command is not None
    )
    if explicit_steering_commands:
        if steering_left_command is None or steering_right_command is None:
            raise ValueError(
                'steering_left_command and steering_right_command must be '
                'provided together'
            )
        command_values = (
            steering_left_command,
            steering_center_command,
            steering_right_command,
        )
        if any(int(value) != value for value in command_values):
            raise ValueError('steering command calibration must be integral')
        left_command, center_command, right_command = (
            int(value) for value in command_values
        )
        if any(
            abs(value) > maximum_steering_step
            for value in (left_command, center_command, right_command)
        ):
            raise ValueError(
                'steering command calibration exceeds maximum_steering_step'
            )
        if left_command == center_command or right_command == center_command:
            raise ValueError(
                'left and right steering commands must differ from centre'
            )
    else:
        center_command = 0

    speed = clamp(float(linear_mps), -maximum_speed_mps, maximum_speed_mps)
    if abs(speed) < 1.0e-4:
        return ArduinoCommand(center_command, 0, 0)

    steering_angle = math.atan(wheelbase_m * float(angular_rps) / speed)
    steering_angle = clamp(
        steering_angle,
        -maximum_steering_angle_rad,
        maximum_steering_angle_rad,
    )
    steering_ratio = clamp(
        steering_direction
        * steering_angle
        / maximum_steering_angle_rad,
        -1.0,
        1.0,
    )
    if explicit_steering_commands:
        if steering_ratio >= 0.0:
            # Positive ROS yaw is a physical left turn.
            steering = round(
                center_command
                + steering_ratio * (left_command - center_command)
            )
        else:
            steering = round(
                center_command
                + (-steering_ratio) * (right_command - center_command)
            )
    else:
        steering = round(steering_ratio * maximum_steering_step)
    steering = int(clamp(
        steering,
        -maximum_steering_step,
        maximum_steering_step,
    ))

    pwm = round(motor_direction * speed / maximum_speed_mps * maximum_pwm)
    pwm = int(clamp(pwm, -maximum_pwm, maximum_pwm))
    return ArduinoCommand(steering, pwm, pwm)


def encode_serial_command(command):
    """Encode the legacy Arduino line protocol."""
    return (
        f's{command.steering}l{command.left_pwm}'
        f'r{command.right_pwm}\n'
    )
