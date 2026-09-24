#!/usr/bin/env python3
# =================================================================================================
# apcbf_twist4.py  —  Twist-space CBF safety filter with SELECTABLE barriers + slack variants.
#
# Subclasses SharedAutonomyController for full baseline perception, disables the baseline blend,
# filters /controller/cmd_vel_raw -> /controller/cmd_vel. The cbf_type parameter selects the core.
#
# HAPTIC OUTPUT (added): publishes nearest-obstacle distance + bearing on /haptic/obstacle
#   (geometry_msgs/Point: x = distance [m], y = bearing [deg], z = 0) so haptic_node_ros2.py in
#   'proximity' mode can render directional vibration. Also publishes the nearest bearing on
#   /haptic/direction (std_msgs/Float32, deg) for the 'cbf' legibility mode (which pairs it with
#   /cbf/h). Both are emitted every perception cycle via the inherited main_loop hook, so haptics
#   work even when the deadman is released / robot is stationary.
# =================================================================================================

import math
import sys

import rclpy
from rclpy.duration import Duration

from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32
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
        super().__init__()                       # full baseline perception (fills self.closestPoints)

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
        self._declare_if_absent('alpha', 2.0)
        self._declare_if_absent('k_rep', 0.375)
        self._declare_if_absent('rho_0', 0.5)
        self._declare_if_absent('rho_cap', 0.3)
        self._declare_if_absent('use_rho_cap', False)
        self._declare_if_absent('d_safe', 0.23)
        self._declare_if_absent('p_slack', 2.0)
        self._declare_if_absent('p_slack_lin', 0.75)
        self._declare_if_absent('p_base', 0.8)
        self._declare_if_absent('eta_w', 0.5)
        self._declare_if_absent('p_elastic_lin', 0.75)
        self._declare_if_absent('p_elastic_quad', 4.0)

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

        # ---- Haptic bearing sign: +1 or -1 to match your robot's left/right convention -----------
        # directional() pans +deg -> RIGHT. atan2(y,x) gives +y = LEFT, so we negate by default so
        # an obstacle on the robot's left buzzes the LEFT actuator. Flip to +1.0 if reversed on hw.
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
        self.haptic_pub     = self.create_publisher(Point, '/haptic/obstacle', 10)   # proximity mode
        self.haptic_dir_pub = self.create_publisher(Float32, '/haptic/direction', 10)  # cbf/legibility

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
            f'  haptic: /haptic/obstacle (x=dist_m,y=bearing_deg) + /haptic/direction\n'
            f'  {self.params}'
        )

    def _declare_if_absent(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    # Override the baseline blend: we DON'T want the APF output, but we DO reuse this hook (called
    # every perception cycle by the inherited main_loop) to publish haptic feedback continuously,
    # independent of whether the operator is driving.
    def publish_finalVelocity_marker(self):
        self._publish_haptic()

    def _joy_deadman_cb(self, msg):
        if self.enable_button < len(msg.buttons):
            self.deadman_held = (msg.buttons[self.enable_button] == 1)
        else:
            self.deadman_held = False

    def _closest_xy(self):
        """Single nearest obstacle point — for 'apf' and 'distance'."""
        if not self.closestPoints:
            return None
        cPoint = min(self.closestPoints, key=lambda p: math.hypot(p.x, p.y))
        return (cPoint.x, cPoint.y)

    def _all_closest_xy(self):
        """All cluster closest points as a list of (x,y) — for every multi variant."""
        return [(p.x, p.y) for p in self.closestPoints]

    def _nearest_dist_bearing(self):
        """Return (distance_m, bearing_deg) of the nearest obstacle, or None if none.
        bearing sign is adjusted by haptic_bearing_sign to match hardware L/R."""
        if not self.closestPoints:
            return None
        nearest = min(self.closestPoints, key=lambda p: math.hypot(p.x, p.y))
        dist_m = math.hypot(nearest.x, nearest.y)
        bearing_deg = self.haptic_bearing_sign * math.degrees(math.atan2(nearest.y, nearest.x))
        return dist_m, bearing_deg

    def _publish_haptic(self):
        """Emit nearest-obstacle distance+bearing for the haptic node. Called every cycle."""
        nb = self._nearest_dist_bearing()
        if nb is None:
            # No obstacle: publish a 'far' distance so the haptic engine outputs silence.
            far = Point()
            far.x = 99.0        # meters -> well beyond max_dst -> force 0
            far.y = 0.0
            far.z = 0.0
            self.haptic_pub.publish(far)
            self.haptic_dir_pub.publish(Float32(data=0.0))
            return
        dist_m, bearing_deg = nb
        p = Point()
        p.x = float(dist_m)
        p.y = float(bearing_deg)
        p.z = 0.0
        self.haptic_pub.publish(p)
        self.haptic_dir_pub.publish(Float32(data=float(bearing_deg)))

    def cmd_callback(self, msg: Twist):
        if not self.deadman_held:
            self.cmd_pub.publish(msg)            # forward as-is (≈zero upstream)
            return

        fn, use_full = self._dispatch[self.cbf_type]
        obstacles = self._all_closest_xy() if use_full else self._closest_xy()
        result = fn(msg.linear.x, msg.linear.y, obstacles, self.params)

        out = Twist()
        out.linear.x  = result.vx
        out.linear.y  = result.vy
        out.angular.x = msg.angular.x
        out.angular.y = msg.angular.y
        out.angular.z = msg.angular.z
        self.cmd_pub.publish(out)

        h_val = result.h if result.h is not None else (1.0 - self.params.delta)
        self.h_pub.publish(Float32(data=float(h_val)))
        self._publish_vfinal_arrow(result.vx, result.vy)
        self.get_logger().info(format_result(result))

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
