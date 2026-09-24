#!/usr/bin/env python3
# =================================================================================================
# apcbf_twist6.py  —  Twist-space CBF safety filter with SELECTABLE barriers + slack variants.
#
# WHAT'S NEW vs apcbf_twist5.py:
#   - Publishes ALL nearby obstacles on /haptic/obstacles (Float32MultiArray) for haptic_node3's
#     multi-obstacle independent L/R rendering (doorway, corridor, multi-object scenarios).
#   - Retains single-obstacle /haptic/obstacle (Point) + /haptic/direction (Float32) for backward
#     compat with haptic_node2 / haptic_node3 in 'proximity_single' and 'cbf' modes.
#   - Throttled logging to avoid terminal back-pressure latency.
#
# Subclasses SharedAutonomyController for full baseline perception, disables the baseline blend,
# filters /controller/cmd_vel_raw -> /controller/cmd_vel. The cbf_type parameter selects the core.
# =================================================================================================

import math
import sys

import rclpy
from rclpy.duration import Duration

from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32, Float32MultiArray
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Joy

from sac3_apf import SharedAutonomyController
from apcbf_core4 import (
    CBFParams,
    cbf_filter,
    cbf_filter_distance,
    cbf_filter_multi,
    cbf_filter_multi_slack,
    cbf_filter_multi_slack_linear,
    cbf_filter_multi_slack_weighted,
    cbf_filter_multi_slack_elastic,
    format_result,
)

VALID_TYPES = (
    'apf',
    'distance',
    'distance_multi',
    'distance_multi_slack',
    'distance_multi_slack_linear',
    'distance_multi_slack_weighted',
    'distance_multi_slack_elastic',
)


class CBFTwistFilter(SharedAutonomyController):

    def __init__(self):
        super().__init__()

        self.declare_parameter('input_cmd_topic', '/controller/cmd_vel_raw')
        self.declare_parameter('output_cmd_topic', '/controller/cmd_vel')
        self.input_cmd_topic  = self.get_parameter('input_cmd_topic').value
        self.output_cmd_topic = self.get_parameter('output_cmd_topic').value
        
        self.declare_parameter('cbf_type', 'distance_multi')
        self.cbf_type = str(self.get_parameter('cbf_type').value)
        if self.cbf_type not in VALID_TYPES:
            raise ValueError(f"cbf_type must be one of {VALID_TYPES}; got '{self.cbf_type}'")

        # ---- CBF tuning --------------------------------------------------------------------------
        self._declare_if_absent('delta', 0.45)
        self._declare_if_absent('alpha', 3)
        self._declare_if_absent('k_rep', 0.375)
        self._declare_if_absent('rho_0', 0.5)
        self._declare_if_absent('rho_cap', 0.25)
        self._declare_if_absent('use_rho_cap', False)
        self._declare_if_absent('d_safe', 0.22)
        self._declare_if_absent('p_slack', 0.000001)
        self._declare_if_absent('p_slack_lin', 0.75)
        self._declare_if_absent('p_base', 0.8)
        self._declare_if_absent('eta_w', 0.5)
        self._declare_if_absent('p_elastic_lin', 0.75)
        self._declare_if_absent('p_elastic_quad', 4.0)
        self._declare_if_absent('rot_block_dist', 0.30)   # block rotation toward side with obstacle closer than this (m)

        self.params = CBFParams(
            delta=float(self.get_parameter('delta').value),
            alpha=float(self.get_parameter('alpha').value),
            k_rep=float(self.get_parameter('k_rep').value),
            rho_0=float(self.get_parameter('rho_0').value),
            rho_cap=float(self.get_parameter('rho_cap').value),
            use_rho_cap=bool(self.get_parameter('use_rho_cap').value),
            d_safe=float(self.get_parameter('d_safe').value),
            p_slack=float(self.get_parameter('p_slack').value),
            p_slack_lin=float(self.get_parameter('p_slack_lin').value),
            p_base=float(self.get_parameter('p_base').value),
            eta_w=float(self.get_parameter('eta_w').value),
            p_elastic_lin=float(self.get_parameter('p_elastic_lin').value),
            p_elastic_quad=float(self.get_parameter('p_elastic_quad').value),
        )

        # ---- Haptic bearing sign -----------------------------------------------------------------
        self.declare_parameter('haptic_bearing_sign', -1.0)
        self.haptic_bearing_sign = float(self.get_parameter('haptic_bearing_sign').value)

        self.declare_parameter('enable_button', 10)
        self.enable_button = int(self.get_parameter('enable_button').value)
        self.deadman_held = False
        self.create_subscription(Joy, '/ps5/joy', self._joy_deadman_cb, 10)

        self.cmd_pub    = self.create_publisher(Twist, self.output_cmd_topic, 10)
        self.h_pub      = self.create_publisher(Float32, '/cbf/h', 10)
        self.marker_pub = self.create_publisher(Marker, '/cbf/vfinal_marker', 10)

        # ---- Haptic feedback publishers ----------------------------------------------------------
        # Single-obstacle (backward compat with haptic_node2 / haptic_node3 proximity_single / cbf)
        self.haptic_pub     = self.create_publisher(Point, '/haptic/obstacle', 10)
        self.haptic_dir_pub = self.create_publisher(Float32, '/haptic/direction', 10)
        # Multi-obstacle (haptic_node3 proximity mode — independent L/R)
        self.haptic_multi_pub = self.create_publisher(Float32MultiArray, '/haptic/obstacles', 10)
        self.haptic_delta_pub = self.create_publisher(Float32, '/haptic/cbf_delta', 10)

        self.create_subscription(Twist, self.input_cmd_topic, self.cmd_callback, 10)

        self.marker_frame = getattr(self, 'base_frame', 'base_link')

        self._dispatch = {
            'apf':                            (cbf_filter,                     False),
            'distance':                       (cbf_filter_distance,            False),
            'distance_multi':                 (cbf_filter_multi,               True),
            'distance_multi_slack':           (cbf_filter_multi_slack,         True),
            'distance_multi_slack_linear':    (cbf_filter_multi_slack_linear,  True),
            'distance_multi_slack_weighted':  (cbf_filter_multi_slack_weighted, True),
            'distance_multi_slack_elastic':   (cbf_filter_multi_slack_elastic, True),
        }

        self.get_logger().info(
            f'CBF Twist filter active.  cbf_type={self.cbf_type}\n'
            f'  IN : {self.input_cmd_topic}\n'
            f'  OUT: {self.output_cmd_topic}\n'
            f'  haptic single: /haptic/obstacle + /haptic/direction\n'
            f'  haptic multi:  /haptic/obstacles (Float32MultiArray)\n'
            f'  {self.params}'
        )

    def _declare_if_absent(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    # Override the baseline blend — publish haptics instead.
    def publish_finalVelocity_marker(self):
        self._publish_haptic()

    def _joy_deadman_cb(self, msg):
        if self.enable_button < len(msg.buttons):
            self.deadman_held = (msg.buttons[self.enable_button] == 1)
        else:
            self.deadman_held = False

    def _closest_xy(self):
        if not self.closestPoints:
            return None
        cPoint = min(self.closestPoints, key=lambda p: math.hypot(p.x, p.y))
        return (cPoint.x, cPoint.y)

    def _all_closest_xy(self):
        return [(p.x, p.y) for p in self.closestPoints]

    def _nearest_dist_bearing(self):
        if not self.closestPoints:
            return None
        nearest = min(self.closestPoints, key=lambda p: math.hypot(p.x, p.y))
        dist_m = math.hypot(nearest.x, nearest.y)
        bearing_deg = self.haptic_bearing_sign * math.degrees(math.atan2(nearest.y, nearest.x))
        return dist_m, bearing_deg

    def _directional_rotation_gate(self, wz_cmd: float) -> float:
        """Scale rotation command based on obstacle proximity on the TURN-INTO side.

        In base_link frame: +y is left, -y is right.
        angular.z > 0 = turning left, angular.z < 0 = turning right.

        Only attenuates when turning TOWARD the side that has a close obstacle.
        Turning away from obstacles (or no obstacles nearby) passes through unchanged.
        """
        if abs(wz_cmd) < 1e-4 or not self.closestPoints:
            return wz_cmd

        rot_block_dist = float(self.get_parameter('rot_block_dist').value)

        # Find minimum obstacle distance on each side (front hemisphere only: x > 0)
        left_min  = float('inf')
        right_min = float('inf')
        for cp in self.closestPoints:
            dist = math.hypot(cp.x, cp.y)
            if dist > self.params.rho_0:
                continue                     # outside influence radius — ignore
            if cp.x < 0:
                continue                     # behind the robot — irrelevant for turning
            if cp.y > 0:
                left_min = min(left_min, dist)
            else:
                right_min = min(right_min, dist)

        # Determine the relevant side based on turn direction
        turning_left = wz_cmd > 0
        side_dist = left_min if turning_left else right_min

        if side_dist >= rot_block_dist:
            return wz_cmd                     # clear on the turn-into side — full rotation

        # Linearly scale: 0 at d_safe, 1 at rot_block_dist
        d_safe = self.params.d_safe
        if side_dist <= d_safe:
            return 0.0                        # too close — block entirely
        scale = (side_dist - d_safe) / (rot_block_dist - d_safe)
        return wz_cmd * scale

    def _publish_haptic(self):
        """Publish haptic data every perception cycle. Emits BOTH single and multi formats."""
        nb = self._nearest_dist_bearing()

        # ---- Single-obstacle (backward compat) ----
        if nb is None:
            far = Point(x=99.0, y=0.0, z=0.0)
            self.haptic_pub.publish(far)
            self.haptic_dir_pub.publish(Float32(data=0.0))
            # Multi: silence sentinel
            multi = Float32MultiArray()
            multi.data = [99.0, 0.0]
            self.haptic_multi_pub.publish(multi)
            return

        dist_m, bearing_deg = nb
        p = Point(x=float(dist_m), y=float(bearing_deg), z=0.0)
        self.haptic_pub.publish(p)
        self.haptic_dir_pub.publish(Float32(data=float(bearing_deg)))

        # ---- Multi-obstacle (for haptic_node3 independent L/R) ----
        multi = Float32MultiArray()
        pairs = []
        for cp in self.closestPoints:
            dist = math.hypot(cp.x, cp.y)
            if dist < self.params.rho_0:
                bearing = self.haptic_bearing_sign * math.degrees(math.atan2(cp.y, cp.x))
                pairs.extend([dist, bearing])
        multi.data = pairs if pairs else [99.0, 0.0]
        self.haptic_multi_pub.publish(multi)

    def cmd_callback(self, msg: Twist):
        if not self.deadman_held:
            self.cmd_pub.publish(msg)
            return

        try:
            fn, use_full = self._dispatch[self.cbf_type]
            obstacles = self._all_closest_xy() if use_full else self._closest_xy()
            result = fn(msg.linear.x, msg.linear.y, obstacles, self.params)
        except Exception as e:
            self.get_logger().error(f'CBF filter crashed: {e}', throttle_duration_sec=1.0)
            self.cmd_pub.publish(msg)
            return

        out = Twist()
        out.linear.x  = result.vx
        out.linear.y  = result.vy
        out.angular.x = msg.angular.x
        out.angular.y = msg.angular.y
        out.angular.z = self._directional_rotation_gate(msg.angular.z)
        self.cmd_pub.publish(out)

        # Block added to publish difference between safe and desired vel.
        import math as _math
        u_des_norm  = _math.hypot(msg.linear.x, msg.linear.y)
        u_safe_norm = _math.hypot(result.vx, result.vy)
        delta = abs(u_des_norm - u_safe_norm)
        self.haptic_delta_pub.publish(Float32(data=float(delta)))

        h_val = result.h if result.h is not None else (1.0 - self.params.delta)
        self.h_pub.publish(Float32(data=float(h_val)))
        self._publish_vfinal_arrow(result.vx, result.vy)
        self.get_logger().info(format_result(result), throttle_duration_sec=1.0)

    def _publish_vfinal_arrow(self, vx, vy):
        m = Marker()
        m.header.frame_id = self.marker_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.type = Marker.ARROW
        m.scale.x, m.scale.y, m.scale.z = 0.05, 0.1, 0.15
        m.color.g = 1.0
        m.color.a = 1.0
        m.pose.orientation.w = 1.0
        m.points = [Point(x=0.0, y=0.0, z=0.0), Point(x=float(vx), y=float(vy), z=0.0)]
        m.lifetime = Duration(seconds=0.2).to_msg()
        self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=sys.argv)
    node = CBFTwistFilter()
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