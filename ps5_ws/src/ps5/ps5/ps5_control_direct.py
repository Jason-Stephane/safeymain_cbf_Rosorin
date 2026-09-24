import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


class PS5ControlDirectNode(Node):
    """
    Joystick -> Twist converter, DIRECT to the motor driver. No safety filter.

        /ps5/joy  ->  THIS NODE  ->  /controller/cmd_vel  ->  odom_publisher -> motors

    Identical conversion logic to the APF/CBF converters, but publishes straight to
    /controller/cmd_vel with nothing in between. Use for raw wheel/teleop testing.
    Only ONE converter should run at a time.
    """

    def __init__(self):
        super().__init__('ps5_control_direct_node')   # distinct node name

        # ========= PS5 CONFIGURATION =========
        self.ENABLE_BUTTON = 10      # R1 - hold to enable movement
        self.MODE_BUTTON = 9        # L1 - toggle control mode
        self.SCALE_LINEAR = 0.3     # m/s
        self.SCALE_ANGULAR = 0.8    # rad/s
        # ======================================

        self.AXIS_LEFT_Y = 1
        self.AXIS_LEFT_X = 0
        self.AXIS_RIGHT_X = 2

        self.current_mode = 1
        self.prev_mode_button = 0
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.SMOOTH_FACTOR = 0.3

        # DIRECT: Joy in, Twist STRAIGHT to the motor driver's input. No filter.
        self.subscription = self.create_subscription(
            Joy, '/ps5/joy', self.joy_callback, 10
        )
        self.publisher = self.create_publisher(
            Twist, '/controller/cmd_vel', 10      # <-- straight to the driver
        )

        self.get_logger().info('PS5 Control Node (DIRECT, no filter) started!')
        self.get_logger().info('Publishing straight to /controller/cmd_vel')

    def joy_callback(self, msg):
        cmd = Twist()

        if msg.buttons[self.ENABLE_BUTTON] == 1 and \
           msg.buttons[self.MODE_BUTTON] == 1 and \
           self.prev_mode_button == 0:
            self.current_mode = 2 if self.current_mode == 1 else 1
            self.get_logger().info(f'Switched to Mode {self.current_mode}')
        self.prev_mode_button = msg.buttons[self.MODE_BUTTON]

        if msg.buttons[self.ENABLE_BUTTON] == 1:
            target_linear = msg.axes[self.AXIS_LEFT_Y] * self.SCALE_LINEAR
            target_angular = msg.axes[self.AXIS_RIGHT_X] * self.SCALE_ANGULAR
            if abs(target_linear) < 0.05:
                target_linear = 0.0
            if abs(target_angular) < 0.05:
                target_angular = 0.0

            # cmd.linear.x = self.prev_linear + self.SMOOTH_FACTOR * (target_linear - self.prev_linear)
            # cmd.angular.z = self.prev_angular + self.SMOOTH_FACTOR * (target_angular - self.prev_angular)
            # self.prev_linear = cmd.linear.x
            # self.prev_angular = cmd.angular.z

            cmd.linear.x = self.prev_linear + self.SMOOTH_FACTOR * (target_linear - self.prev_linear)
            cmd.angular.z = self.prev_angular + self.SMOOTH_FACTOR * (target_angular - self.prev_angular)

            # Snap near-zero smoothing tails to exact zero
            if abs(cmd.linear.x) < 1e-4:
                cmd.linear.x = 0.0
            if abs(cmd.angular.z) < 1e-4:
                cmd.angular.z = 0.0

            self.prev_linear = cmd.linear.x
            self.prev_angular = cmd.angular.z

            if self.current_mode == 2:
                target_strafe = msg.axes[self.AXIS_LEFT_X] * self.SCALE_LINEAR
                if abs(target_strafe) < 0.05:
                    target_strafe = 0.0
                cmd.linear.y = target_strafe
        else:
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.0
            self.prev_linear = 0.0
            self.prev_angular = 0.0

        self.publisher.publish(cmd)


def main():
    rclpy.init()
    node = PS5ControlDirectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()