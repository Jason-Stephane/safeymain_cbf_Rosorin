#!/usr/bin/env python3
# =================================================================================================
# haptic_node5.py  —  ROS2 shell around haptic_core4 with multi-obstacle support.
#
# WHAT'S NEW vs haptic_node4.py:
#   - cbf mode uses /haptic/cbf_delta topic (matches apcbf_twist7 publisher).
#
# source = 'proximity' subscribes ONLY to /haptic/obstacles (Float32MultiArray) for
#   independent L/R rendering from multiple obstacles.
#
# source = 'proximity_single' preserves the original single-obstacle behavior
#   (subscribes to /haptic/obstacle Point, uses set_targets with panning).
#
# source = 'cbf' uses CBF disagreement delta for intensity.
#
# Run:
#   python3 haptic_node5.py --ros-args -p source:=proximity          # multi-obstacle (default)
#   python3 haptic_node5.py --ros-args -p source:=proximity_single   # single-obstacle (legacy)
#   python3 haptic_node5.py --ros-args -p source:=cbf                # legibility mode
# =================================================================================================

import os
import sys

os.environ.setdefault(
    "PULSE_SINK",
    "alsa_output.usb-Sony_Interactive_Entertainment_Wireless_Controller-00.analog-surround-40",
)

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Point, Twist

from haptic_core5 import HapticAudioEngine


class HapticNodeROS2(Node):
    def __init__(self):
        super().__init__("haptic_audio_node")

        # ---- parameters ----
        self.declare_parameter("source", "proximity")    # 'proximity' | 'proximity_single' | 'cbf'
        self.declare_parameter("device", "pulse")
        self.declare_parameter("d_safe", 0.3)
        self.declare_parameter("max_dst_m", 0.7)
        self.declare_parameter("min_dst_m", 0.3)
        # ---- rear-warning beep ----
        self.declare_parameter("cmd_vel_topic", "/controller/cmd_vel")
        self.declare_parameter("rear_center_deg", 180.0)
        self.declare_parameter("rear_halfwidth_deg", 60.0)
        self.declare_parameter("beep_amp", 0.4)
        self.declare_parameter("reverse_thresh", 0.01)

        self.source = str(self.get_parameter("source").value)
        self.d_safe = float(self.get_parameter("d_safe").value)
        self.rear_center = float(self.get_parameter("rear_center_deg").value)
        self.rear_halfwidth = float(self.get_parameter("rear_halfwidth_deg").value)
        self.beep_amp = float(self.get_parameter("beep_amp").value)
        self.reverse_thresh = float(self.get_parameter("reverse_thresh").value)
        self._vx = 0.0

        self.declare_parameter("force_func", 2)
        self.declare_parameter("alpha", 0.5)

        min_d = float(self.get_parameter("min_dst_m").value)
        max_d = float(self.get_parameter("max_dst_m").value)

        self.engine = HapticAudioEngine(
            device=str(self.get_parameter("device").value),
            haptic_channels=(2, 3),
            freq=60.0,
            master=0.4,
            min_dst=min_d,
            max_dst=max_d,
            force_func=int(self.get_parameter("force_func").value),
            alpha=float(self.get_parameter("alpha").value),
            smooth=0.0008,
        )

        self._direction_deg = 0.0

        # velocity command feed for the reverse gate (all modes)
        self.create_subscription(
            Twist, str(self.get_parameter("cmd_vel_topic").value), self._cmd_vel_cb, 1)

        # ---- source selection ----
        if self.source == "proximity":
            # MULTI-OBSTACLE: independent L/R from all nearby obstacles.
            self.create_subscription(
                Float32MultiArray, "/haptic/obstacles", self._multi_obstacle_cb, 1)
            self.get_logger().info(
                "Haptic source = PROXIMITY (multi-obstacle)\n"
                "  topic: /haptic/obstacles (Float32MultiArray: [d1,b1,d2,b2,...])")

        elif self.source == "proximity_single":
            # SINGLE-OBSTACLE (legacy): panned L/R from nearest obstacle only.
            self.create_subscription(Point, "/haptic/obstacle", self._proximity_single_cb, 1)
            self.get_logger().info(
                "Haptic source = PROXIMITY_SINGLE\n"
                "  topic: /haptic/obstacle (Point: x=dist_m, y=dir_deg)")

        elif self.source == "cbf":
            self.declare_parameter("max_delta", 0.9)
            self._max_delta = float(self.get_parameter("max_delta").value)
            self._latest_delta = 0.0
            # Subscribe to CBF delta (correction magnitude) from apcbf_twist7
            self.create_subscription(Float32, "/haptic/cbf_delta",  self._cbf_delta_cb, 1)
            self.create_subscription(Float32, "/haptic/direction",  self._direction_cb, 1)
            self.get_logger().info("Haptic source = CBF (disagreement intensity)")

        else:
            raise ValueError(
                f"source must be 'proximity', 'proximity_single', or 'cbf'; got '{self.source}'")

    # ============================ multi-obstacle (NEW default) ============================
    def _multi_obstacle_cb(self, msg: Float32MultiArray):
        """Process [dist1, bearing1, dist2, bearing2, ...] for independent L/R haptics."""
        data = list(msg.data)
        if len(data) < 2:
            self.engine.set_multi_targets([])
            return

        obstacles = []
        nearest_behind = None
        for i in range(0, len(data) - 1, 2):
            dist, bearing = data[i], data[i + 1]
            obstacles.append((dist, bearing))
            # Track nearest rear obstacle for beep
            if abs(self._angle_from_rear(bearing)) <= self.rear_halfwidth:
                if nearest_behind is None or dist < nearest_behind[0]:
                    nearest_behind = (dist, bearing)

        self.engine.set_multi_targets(obstacles)

        # Rear beep based on nearest behind-obstacle
        if nearest_behind is not None:
            self._update_rear_beep(nearest_behind[0], nearest_behind[1])
        else:
            self.engine.set_beep(0.0)

    # ============================ single-obstacle (legacy) ============================
    def _proximity_single_cb(self, msg: Point):
        """Single nearest obstacle → panned L/R (original behavior)."""
        distance = msg.x
        direction_deg = msg.y
        self.engine.set_targets(distance, direction_deg)
        self._update_rear_beep(distance, direction_deg)

    # ============================ rear-warning beep ============================
    def _cmd_vel_cb(self, msg: Twist):
        self._vx = float(msg.linear.x)

    def _angle_from_rear(self, direction_deg):
        return ((direction_deg - self.rear_center + 180.0) % 360.0) - 180.0

    def _update_rear_beep(self, distance, direction_deg):
        force = self.engine.get_force(distance)
        behind = abs(self._angle_from_rear(direction_deg)) <= self.rear_halfwidth
        reversing = self._vx < -self.reverse_thresh
        if behind and reversing and force > 0.0:
            self.engine.set_beep(self.beep_amp, rate=1.5 + 10.0 * force)
        else:
            self.engine.set_beep(0.0)

    # ============================ cbf / legibility mode ============================
    def _cbf_delta_cb(self, msg: Float32):
        self._latest_delta = float(msg.data)
        self.engine.set_disagreement(
            self._latest_delta, 0.0,
            self._direction_deg, self._max_delta)

    def _direction_cb(self, msg: Float32):
        self._direction_deg = float(msg.data)

    # ============================ lifecycle ============================
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