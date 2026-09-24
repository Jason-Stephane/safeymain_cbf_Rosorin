#!/usr/bin/env python3
# =================================================================================================
# haptic_node_ros2.py  —  ROS2 (rclpy) shell around haptic_audio_core.
#
# For the HIWONDER / CBF stack. Same shared audio engine as the ROS1 shell; only the ROS glue and
# the input message differ. Contains NO audio logic.
#
# INPUT SIGNAL (choose via the 'source' parameter):
#   source = 'proximity'  -> subscribe to a Point on /haptic/obstacle where
#                              x = distance (m), y = direction (deg).
#                            (Publish this from your CBF filter's nearest-obstacle info.)
#   source = 'cbf'        -> LEGIBILITY MODE: subscribe to /cbf/h (Float32) and map the barrier
#                            value to cue intensity, so the vibration reflects the FILTER's
#                            intervention ("the safety layer is acting"), not raw proximity.
#                            (Direction defaults to 0 unless you also publish /haptic/direction.)
#
# The two robots publish different perception, so the shell owns the message unpacking; the core is
# identical to the ROS1 side.
#
# Run:  ros2 run <pkg> haptic_node_ros2      (or python3 haptic_node_ros2.py)
#       python3 haptic_node_ros2.py --ros-args -p source:=proximity
#       python3 haptic_node_ros2.py --ros-args -p source:=cbf -p d_safe:=0.3
# =================================================================================================

import os
import sys

# DualSense USB audio sink. Machine-specific -> set in the shell. VERIFY the exact name on the
# Hiwonder machine with:  python3 -c "import sounddevice as sd; print(sd.query_devices())"
os.environ.setdefault(
    "PULSE_SINK",
    "alsa_output.usb-Sony_Interactive_Entertainment_Wireless_Controller-00.analog-surround-40",
)

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Point, Twist

from haptic_core import HapticAudioEngine


class HapticNodeROS2(Node):
    def __init__(self):
        super().__init__("haptic_audio_node")

        # ---- parameters ----
        self.declare_parameter("source", "proximity")   # 'proximity' | 'cbf'
        self.declare_parameter("device", "pulse")
        self.declare_parameter("d_safe", 0.3)            # used by 'cbf' mode to map h -> distance
        self.declare_parameter("max_dst_m", 0.6)         # m: cue begins
        self.declare_parameter("min_dst_m", 0.35)         # m: cue saturates
        # ---- rear-warning beep (speaker) ----
        self.declare_parameter("cmd_vel_topic", "/controller/cmd_vel")  # what the robot EXECUTES
        self.declare_parameter("rear_center_deg", 180.0)  # bearing that means "directly behind"
        self.declare_parameter("rear_halfwidth_deg", 60.0)  # obstacle counts as behind within +/- this
        self.declare_parameter("beep_amp", 0.7)           # speaker loudness 0..1
        self.declare_parameter("reverse_thresh", 0.01)    # m/s: vx < -thresh counts as reversing
        self.source = str(self.get_parameter("source").value)
        self.d_safe = float(self.get_parameter("d_safe").value)
        self.rear_center = float(self.get_parameter("rear_center_deg").value)
        self.rear_halfwidth = float(self.get_parameter("rear_halfwidth_deg").value)
        self.beep_amp = float(self.get_parameter("beep_amp").value)
        self.reverse_thresh = float(self.get_parameter("reverse_thresh").value)
        self._vx = 0.0                                     # latest commanded linear.x
        self.declare_parameter("force_func", 2) #added to test the different force functions
        self.declare_parameter("alpha", 1.0)

        min_d = float(self.get_parameter("min_dst_m").value)
        max_d = float(self.get_parameter("max_dst_m").value)

        self.engine = HapticAudioEngine(
            device=str(self.get_parameter("device").value),
            haptic_channels=(2, 3),
            freq=60.0,
            master=0.8,
            min_dst=min_d,
            max_dst=max_d,
            force_func=int(self.get_parameter("force_func").value),
            alpha=float(self.get_parameter("alpha").value),
            smooth=0.0008,
        )

        self._direction_deg = 0.0   # updated by /haptic/direction if provided

        # velocity command feed for the reverse gate (both modes)
        self.create_subscription(
            Twist, str(self.get_parameter("cmd_vel_topic").value), self._cmd_vel_cb, 1)

        if self.source == "proximity":
            # x = distance (m), y = direction (deg). Publish from the CBF nearest-obstacle info.
            self.create_subscription(Point, "/haptic/obstacle", self._proximity_cb, 1)
            self.get_logger().info("Haptic source = PROXIMITY  (/haptic/obstacle: x=dist_m, y=dir_deg)")
        elif self.source == "cbf":
            # LEGIBILITY: barrier value -> cue intensity. Optional separate direction topic.
            self.create_subscription(Float32, "/cbf/h", self._cbf_cb, 1)
            self.create_subscription(Float32, "/haptic/direction", self._direction_cb, 1)
            self.get_logger().info("Haptic source = CBF (legibility)  (/cbf/h -> intensity)")
        else:
            raise ValueError(f"source must be 'proximity' or 'cbf'; got '{self.source}'")

    # ---- proximity mode: distance+direction straight to the engine ----
    def _proximity_cb(self, msg: Point):
        distance = msg.x      # m: removed the dm conversion
        direction_deg = msg.y
        self.engine.set_targets(distance, direction_deg)
        self._update_rear_beep(distance, direction_deg)

    # ---- rear-warning beep: obstacle behind AND commanding reverse ----
    def _cmd_vel_cb(self, msg: Twist):
        self._vx = float(msg.linear.x)

    def _angle_from_rear(self, direction_deg):
        """Smallest signed angle between the obstacle bearing and 'directly behind'."""
        return ((direction_deg - self.rear_center + 180.0) % 360.0) - 180.0

    def _update_rear_beep(self, distance, direction_deg):
        force = self.engine.get_force(distance)          # reuse the same closeness curve
        behind = abs(self._angle_from_rear(direction_deg)) <= self.rear_halfwidth
        reversing = self._vx < -self.reverse_thresh
        if behind and reversing and force > 0.0:
            # repetition rate encodes urgency: ~2 Hz at the outer edge -> ~12 Hz at saturation
            self.engine.set_beep(self.beep_amp, rate=1.5 + 10.0 * force)
        else:
            self.engine.set_beep(0.0)

    # ---- cbf/legibility mode: map barrier h -> an effective distance the engine understands ----
    def _cbf_cb(self, msg: Float32):
        # h = rho - d_safe  =>  rho = h + d_safe, so
        # the cue rises as the filter nears/violates its barrier. Direction from /haptic/direction.
        rho_m = float(msg.data) + self.d_safe
        distance = rho_m
        self.engine.set_targets(distance, self._direction_deg)

    def _direction_cb(self, msg: Float32):
        self._direction_deg = float(msg.data)

    def start(self):
        self.engine.start()

    def shutdown(self):
        self.engine.stop()


def main(args=None):
    rclpy.init(args=sys.argv)
    node = HapticNodeROS2()
    node.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
