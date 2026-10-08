import math

from vehicle_interface_pkg.command_conversion import encode_serial_command
from vehicle_interface_pkg.command_conversion import velocity_to_arduino_command


COMMON = {
    'wheelbase_m': 0.8,
    'maximum_steering_angle_rad': math.radians(20.0),
    'maximum_steering_step': 7,
    'steering_left_command': -7,
    'steering_center_command': 0,
    'steering_right_command': 7,
    'maximum_speed_mps': 1.0,
    'maximum_pwm': 100,
}


def test_stop_maps_to_zero_command():
    command = velocity_to_arduino_command(0.0, 2.0, **COMMON)
    assert command.steering == 0
    assert command.left_pwm == 0
    assert command.right_pwm == 0


def test_straight_speed_maps_to_equal_pwm():
    command = velocity_to_arduino_command(0.5, 0.0, **COMMON)
    assert command.steering == 0
    assert command.left_pwm == 50
    assert command.right_pwm == 50


def test_left_yaw_maps_to_reference_firmware_negative_steering():
    command = velocity_to_arduino_command(1.0, 0.3, **COMMON)
    assert command.steering < 0
    assert command.left_pwm == 100
    assert command.right_pwm == 100


def test_limits_and_protocol():
    command = velocity_to_arduino_command(5.0, -20.0, **COMMON)
    assert command.steering == 7
    assert command.left_pwm == 100
    assert command.right_pwm == 100
    assert encode_serial_command(command) == 's7l100r100\n'


def test_asymmetric_commands_interpolate_around_configured_centre():
    parameters = dict(COMMON)
    parameters.update({
        'steering_left_command': -6,
        'steering_center_command': 1,
        'steering_right_command': 7,
    })
    straight = velocity_to_arduino_command(0.5, 0.0, **parameters)
    stopped = velocity_to_arduino_command(0.0, 1.0, **parameters)
    full_left = velocity_to_arduino_command(1.0, 20.0, **parameters)
    full_right = velocity_to_arduino_command(1.0, -20.0, **parameters)
    assert straight.steering == 1
    assert stopped.steering == 1
    assert full_left.steering == -6
    assert full_right.steering == 7


def test_legacy_symmetric_mapping_remains_available():
    parameters = dict(COMMON)
    parameters.pop('steering_left_command')
    parameters.pop('steering_center_command')
    parameters.pop('steering_right_command')
    command = velocity_to_arduino_command(1.0, 20.0, **parameters)
    assert command.steering == 7
