import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


class PS5ControlCBFNode(Node):
    """
    Joystick -> Twist converter for the CBF pipeline.

    Identical conversion logic to ps5_controller.py, but wired for the CBF chain:
        /ps5/joy  ->  THIS NODE  ->  /controller/cmd_vel_raw  ->  CBF filter  ->  /controller/cmd_vel

    Distinct node name (ps5_control_cbf_node) so it is discoverable separately from the APF
    converter. Only ONE converter should run at a time.
    """

    def __init__(self):
        super().__init__('ps5_control_cbf_node')   # distinct from the APF 'ps5_control_node'

        # ========= PS5 CONFIGURATION =========
        self.ENABLE_BUTTON = 5      # R1 - hold to enable movement
        self.MODE_BUTTON = 4        # L1 - toggle control mode
        self.SCALE_LINEAR = 0.3     # Maximum linear speed (m/s)
        self.SCALE_ANGULAR = 0.8    # Maximum angular speed (rad/s)
        # ======================================

        # Axes mapping
        self.AXIS_LEFT_Y = 1        # Left stick Y - forward/backward
        self.AXIS_LEFT_X = 0        # Left stick X - strafe (mode 2 only)
        self.AXIS_RIGHT_X = 2       # Right stick X - rotate

        self.current_mode = 1
        self.prev_mode_button = 0

        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.SMOOTH_FACTOR = 0.3

        # CBF pipeline: Joy in from the stick, Twist out to the CBF filter's input topic.
        self.subscription = self.create_subscription(
            Joy,
            '/ps5/joy',
            self.joy_callback,
            10
        )

        self.publisher = self.create_publisher(
            Twist,
            '/controller/cmd_vel_raw',           # <-- CBF filter reads this (NOT /controller/cmd_vel)
            10
        )

        self.get_logger().info('PS5 Control Node (CBF pipeline) started!')
        self.get_logger().info('Publishing to /controller/cmd_vel_raw')
        self.get_logger().info('Mode 1: Left stick forward/back + Right stick rotate')
        self.get_logger().info('Press R1+L1 to toggle mode')

    def joy_callback(self, msg):
        cmd = Twist()

        # Detect L1 press while holding R1 (toggle mode)
        if msg.buttons[self.ENABLE_BUTTON] == 1 and \
           msg.buttons[self.MODE_BUTTON] == 1 and \
           self.prev_mode_button == 0:
            if self.current_mode == 1:
                self.current_mode = 2
                self.get_logger().info('Switched to Mode 2: Left stick forward/back/strafe + Right stick rotate')
            else:
                self.current_mode = 1
                self.get_logger().info('Switched to Mode 1: Left stick forward/back + Right stick rotate')
        self.prev_mode_button = msg.buttons[self.MODE_BUTTON]

        # Only move when R1 is held
        if msg.buttons[self.ENABLE_BUTTON] == 1:
            target_linear = msg.axes[self.AXIS_LEFT_Y] * self.SCALE_LINEAR
            target_angular = msg.axes[self.AXIS_RIGHT_X] * self.SCALE_ANGULAR

            if abs(target_linear) < 0.05:
                target_linear = 0.0
            if abs(target_angular) < 0.05:
                target_angular = 0.0

            # Smooth filter
            cmd.linear.x = self.prev_linear + self.SMOOTH_FACTOR * (target_linear - self.prev_linear)
            cmd.angular.z = self.prev_angular + self.SMOOTH_FACTOR * (target_angular - self.prev_angular)

            self.prev_linear = cmd.linear.x
            self.prev_angular = cmd.angular.z

            # Mode 2: add strafe with left stick X
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
    node = PS5ControlCBFNode()
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