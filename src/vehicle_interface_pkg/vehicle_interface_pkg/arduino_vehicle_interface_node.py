"""Send the final, safety-gated ROS velocity command to an Arduino vehicle."""

import math

from geometry_msgs.msg import Twist
from interfaces_pkg.msg import MotionCommand
import rclpy
from rclpy.node import Node

from .command_conversion import encode_serial_command
from .command_conversion import velocity_to_arduino_command

try:
    import serial
except ImportError:  # pragma: no cover - reported clearly when hardware starts
    serial = None


class ArduinoVehicleInterfaceNode(Node):
    """Bridge the final ``/cmd_vel`` boundary to the USB serial controller."""

    def __init__(self):
        super().__init__('arduino_vehicle_interface_node')

        self.declare_parameter('input_command_topic', '/cmd_vel')
        self.declare_parameter(
            'debug_command_topic', '/vehicle/arduino_command'
        )
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('serial_timeout_s', 0.1)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('wheelbase_m', 1.0)
        self.declare_parameter('maximum_steering_angle_deg', 30.0)
        self.declare_parameter('maximum_steering_step', 7)
        self.declare_parameter('steering_left_command', -7)
        self.declare_parameter('steering_center_command', 0)
        self.declare_parameter('steering_right_command', 7)
        self.declare_parameter('maximum_speed_mps', 1.0)
        self.declare_parameter('maximum_pwm', 100)
        self.declare_parameter('steering_direction', 1.0)
        self.declare_parameter('motor_direction', 1.0)
        self.declare_parameter('terminal_log_period_s', 1.0)

        self.serial_port_name = str(self.get_parameter('serial_port').value)
        self.baud_rate = int(self.get_parameter('baud_rate').value)
        self.serial_timeout_s = float(
            self.get_parameter('serial_timeout_s').value
        )
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.wheelbase_m = float(self.get_parameter('wheelbase_m').value)
        self.maximum_steering_angle_rad = math.radians(float(
            self.get_parameter('maximum_steering_angle_deg').value
        ))
        self.maximum_steering_step = int(
            self.get_parameter('maximum_steering_step').value
        )
        self.steering_left_command = int(
            self.get_parameter('steering_left_command').value
        )
        self.steering_center_command = int(
            self.get_parameter('steering_center_command').value
        )
        self.steering_right_command = int(
            self.get_parameter('steering_right_command').value
        )
        self.maximum_speed_mps = float(
            self.get_parameter('maximum_speed_mps').value
        )
        self.maximum_pwm = int(self.get_parameter('maximum_pwm').value)
        self.steering_direction = float(
            self.get_parameter('steering_direction').value
        )
        self.motor_direction = float(
            self.get_parameter('motor_direction').value
        )
        self.terminal_log_period_s = max(
            0.1, float(self.get_parameter('terminal_log_period_s').value)
        )

        # Validate all conversion parameters before opening vehicle hardware.
        velocity_to_arduino_command(
            0.0,
            0.0,
            wheelbase_m=self.wheelbase_m,
            maximum_steering_angle_rad=self.maximum_steering_angle_rad,
            maximum_steering_step=self.maximum_steering_step,
            maximum_speed_mps=self.maximum_speed_mps,
            maximum_pwm=self.maximum_pwm,
            steering_direction=self.steering_direction,
            motor_direction=self.motor_direction,
            steering_left_command=self.steering_left_command,
            steering_center_command=self.steering_center_command,
            steering_right_command=self.steering_right_command,
        )

        self.serial_connection = None
        if self.dry_run:
            self.get_logger().warning(
                'Arduino interface is in dry-run mode; no serial data is sent'
            )
        else:
            if serial is None:
                raise RuntimeError(
                    'pyserial is unavailable; install the python3-serial package'
                )
            self.serial_connection = serial.Serial(
                self.serial_port_name,
                self.baud_rate,
                timeout=self.serial_timeout_s,
            )
            self.get_logger().info(
                'Opened Arduino serial port {} at {} bps'.format(
                    self.serial_port_name,
                    self.baud_rate,
                )
            )

        input_topic = str(self.get_parameter('input_command_topic').value)
        debug_topic = str(self.get_parameter('debug_command_topic').value)
        self.command_subscription = self.create_subscription(
            Twist,
            input_topic,
            self._on_command,
            1,
        )
        self.debug_publisher = self.create_publisher(
            MotionCommand,
            debug_topic,
            1,
        )
        self.last_log_time_ns = 0

        self.get_logger().info(
            'Vehicle interface ready: {} -> Arduino, wheelbase={:.3f} m, '
            'steering commands left/centre/right={}/{}/{}, physical '
            'limit=+/-{:.1f} deg, speed=+/-{:.3f} m/s -> PWM +/-{}'.format(
                input_topic,
                self.wheelbase_m,
                self.steering_left_command,
                self.steering_center_command,
                self.steering_right_command,
                math.degrees(self.maximum_steering_angle_rad),
                self.maximum_speed_mps,
                self.maximum_pwm,
            )
        )

    def _convert(self, message):
        return velocity_to_arduino_command(
            message.linear.x,
            message.angular.z,
            wheelbase_m=self.wheelbase_m,
            maximum_steering_angle_rad=self.maximum_steering_angle_rad,
            maximum_steering_step=self.maximum_steering_step,
            maximum_speed_mps=self.maximum_speed_mps,
            maximum_pwm=self.maximum_pwm,
            steering_direction=self.steering_direction,
            motor_direction=self.motor_direction,
            steering_left_command=self.steering_left_command,
            steering_center_command=self.steering_center_command,
            steering_right_command=self.steering_right_command,
        )

    def _send(self, command, *, publish_debug=True):
        encoded = encode_serial_command(command)
        if self.serial_connection is not None:
            self.serial_connection.write(encoded.encode('ascii'))

        # SIGINT can invalidate the ROS context before ``finally`` runs.  The
        # serial stop must still be sent, but publishing after that point would
        # raise RCLError and make an otherwise clean shutdown look like a
        # process failure.
        if publish_debug:
            debug = MotionCommand()
            debug.steering = command.steering
            debug.left_speed = command.left_pwm
            debug.right_speed = command.right_pwm
            self.debug_publisher.publish(debug)

    def _on_command(self, message):
        command = self._convert(message)
        self._send(command, publish_debug=rclpy.ok())

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_log_time_ns >= int(
            self.terminal_log_period_s * 1.0e9
        ):
            self.get_logger().info(
                'cmd_vel=({:.3f} m/s, {:.3f} rad/s) -> '
                'steering={} left_pwm={} right_pwm={}'.format(
                    message.linear.x,
                    message.angular.z,
                    command.steering,
                    command.left_pwm,
                    command.right_pwm,
                )
            )
            self.last_log_time_ns = now_ns

    def send_stop(self):
        """Send a normal-shutdown stop without implementing a watchdog yet."""
        command = velocity_to_arduino_command(
            0.0,
            0.0,
            wheelbase_m=self.wheelbase_m,
            maximum_steering_angle_rad=self.maximum_steering_angle_rad,
            maximum_steering_step=self.maximum_steering_step,
            maximum_speed_mps=self.maximum_speed_mps,
            maximum_pwm=self.maximum_pwm,
            steering_direction=self.steering_direction,
            motor_direction=self.motor_direction,
            steering_left_command=self.steering_left_command,
            steering_center_command=self.steering_center_command,
            steering_right_command=self.steering_right_command,
        )
        self._send(command, publish_debug=False)

    def close_serial(self):
        """Close the Arduino connection if it was opened."""
        if self.serial_connection is not None:
            self.serial_connection.close()
            self.serial_connection = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ArduinoVehicleInterfaceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.send_stop()
            finally:
                node.close_serial()
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
