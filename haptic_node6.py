"""
python3 haptic_node6.py --ros-args \
  -p source:=bpsdf \
  -p calibration_file:=haptic_calibration.json \
  -p participant_id:=P01
"""

import json
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
        self.declare_parameter("source", "bpsdf")    # 'proximity' | 'proximity_single' | 'bpsdf' | 'cbf'
        self.declare_parameter("device", "pulse")
        self.declare_parameter("d_safe", 0.3)
        self.declare_parameter("max_dst_m", 0.7)
        self.declare_parameter("min_dst_m", 0.3)
        self.declare_parameter("master", 0.1)
        # ---- rear-warning beep ----
        self.declare_parameter("cmd_vel_topic", "/controller/cmd_vel")
        self.declare_parameter("rear_center_deg", 180.0)
        self.declare_parameter("rear_halfwidth_deg", 60.0)
        self.declare_parameter("beep_amp", 0.2)
        self.declare_parameter("reverse_thresh", 0.01)
        # ---- per-user calibration ----
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("participant_id", "")

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
        master = float(self.get_parameter("master").value)

        # ---- Load per-user calibration if provided ----
        self._cal_low = 0.0     # intensity floor (maps to user's perception threshold)
        self._cal_high = 1.0    # intensity ceiling (maps to user's comfort limit)
        cal_file = str(self.get_parameter("calibration_file").value)
        pid = str(self.get_parameter("participant_id").value)
        if cal_file and pid:
            master = self._load_calibration(cal_file, pid, master)

        self.engine = HapticAudioEngine(
            device=str(self.get_parameter("device").value),
            haptic_channels=(2, 3),
            freq=60.0,
            master=master,
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
            self.create_subscription(
                Float32MultiArray, "/haptic/obstacles", self._multi_obstacle_cb, 1)
            self.get_logger().info(
                "Haptic source = PROXIMITY (multi-obstacle)\n"
                "  topic: /haptic/obstacles (Float32MultiArray: [d1,b1,d2,b2,...])")

        elif self.source == "proximity_single":
            self.create_subscription(Point, "/haptic/obstacle", self._proximity_single_cb, 1)
            self.get_logger().info(
                "Haptic source = PROXIMITY_SINGLE\n"
                "  topic: /haptic/obstacle (Point: x=dist_m, y=dir_deg)")

        elif self.source == "bpsdf":
            # Shape-aware: use barrier value h directly (already encodes rectangle)
            self.declare_parameter("h_max", 0.15)       # h above this → no vibration
            self._h_max = float(self.get_parameter("h_max").value)

            # ---- flanked (left+right) ping-pong pulsing ----
            # Reuses the engine's existing sample-accurate "corridor" cue (the same
            # one set_multi_targets uses for proximity mode) instead of a ROS timer,
            # so there's no timer jitter and only one pulsing implementation to tune.
            self.declare_parameter("flank_pulse_rate_hz", 1.5)   # full L->R cycles/sec when flanked
            self.declare_parameter("flank_pulse_gap", 0.10)      # silent fraction of each half-cycle
            self.declare_parameter("flank_cone_deg", 100.0)      # +/- degrees from front counted as "side"
            self.declare_parameter("flank_front_deadzone_deg", 35.0)  # matches engine's FRONT_HALF
            self.declare_parameter("flank_max_dist_m", -1.0)     # <=0 -> reuse max_dst_m
            self._flank_cone = float(self.get_parameter("flank_cone_deg").value)
            self._flank_deadzone = float(self.get_parameter("flank_front_deadzone_deg").value)
            flank_dist_param = float(self.get_parameter("flank_max_dist_m").value)
            self._flank_max_dist = flank_dist_param if flank_dist_param > 0 else max_d

            self._flanked = False           # true when obstacles present on both L and R
            self._latest_intensity = 0.0    # last calibrated 0-1 intensity from /cbf/h

            # self.engine already exists at this point (built earlier in __init__);
            # configure its corridor timing directly from these params.
            self._flank_pulse_rate = float(self.get_parameter("flank_pulse_rate_hz").value)
            self._flank_pulse_gap = float(self.get_parameter("flank_pulse_gap").value)
            self.engine._corridor_rate = self._flank_pulse_rate
            self.engine._corridor_gap = self._flank_pulse_gap

            self.create_subscription(Float32, "/cbf/h",              self._bpsdf_h_cb, 1)
            self.create_subscription(Float32, "/haptic/direction",   self._direction_cb, 1)
            # Also subscribe to obstacles: rear beep gate + left/right flank detection
            self.create_subscription(
                Float32MultiArray, "/haptic/obstacles", self._obstacles_cb, 1)

            self.get_logger().info(
                f"Haptic source = BPSDF (barrier h → intensity)\n"
                f"  h_max={self._h_max:.3f}  (vibration starts when h drops below this)\n"
                f"  flank pulse: rate={self._flank_pulse_rate:.1f}Hz gap={self._flank_pulse_gap:.2f} "
                f"cone=±{self._flank_cone:.0f}deg deadzone=±{self._flank_deadzone:.0f}deg "
                f"max_dist={self._flank_max_dist:.2f}m\n"
                f"  topics: /cbf/h, /haptic/direction, /haptic/obstacles (rear beep + flank)")

        elif self.source == "cbf":
            self.declare_parameter("max_delta", 0.15)
            self._max_delta = float(self.get_parameter("max_delta").value)
            self._latest_delta = 0.0
            self.create_subscription(Float32, "/haptic/cbf_delta",  self._cbf_delta_cb, 1)
            self.create_subscription(Float32, "/haptic/direction",  self._direction_cb, 1)
            self.get_logger().info(
                f"Haptic source = CBF (disagreement intensity)\n"
                f"  max_delta={self._max_delta:.3f}")

        else:
            raise ValueError(
                f"source must be 'proximity', 'proximity_single', 'bpsdf', or 'cbf'; "
                f"got '{self.source}'")

    # ============================ calibration ============================
    def _load_calibration(self, cal_file, pid, master):
        """
        Load per-participant calibration from JSON.
        Expected format (written by haptic_calibrate.py):
        {
          "P01": {"master_low": 0.03, "master_high": 0.18, "master_max": 0.25, "freq": 60.0},
          "P02": { ... }
        }
        master_low/high are the raw amplitudes at the user's perception threshold
        and comfort ceiling. master_max is the amplitude ceiling used during
        calibration — adopted as the engine's master so the calibrated range
        maps correctly without passing master separately.

        Returns the master value to use for the engine.
        """
        try:
            with open(cal_file, 'r') as f:
                cal = json.load(f)
            if pid in cal:
                entry = cal[pid]
                # Use the master_max from calibration so the range maps correctly
                cal_master = entry.get("master_max", master)
                if cal_master > 0:
                    master = cal_master
                self._cal_low  = entry.get("master_low", 0.0) / max(1e-6, master)
                self._cal_high = entry.get("master_high", master) / max(1e-6, master)
                self._cal_high = max(self._cal_high, self._cal_low + 0.01)
                self.get_logger().info(
                    f"Calibration loaded for {pid}: "
                    f"master={master:.4f}  "
                    f"low={self._cal_low:.3f} high={self._cal_high:.3f} (normalized)")
            else:
                self.get_logger().warn(f"No calibration for '{pid}' in {cal_file}; using defaults")
        except Exception as e:
            self.get_logger().error(f"Could not load calibration file: {e}")
        return master

    def _apply_calibration(self, intensity_01):
        """
        Map raw 0-1 intensity into the user's calibrated range.
        0.0 → cal_low (just perceptible), 1.0 → cal_high (comfort limit).
        """
        return self._cal_low + intensity_01 * (self._cal_high - self._cal_low)

    # ============================ bpsdf mode (NEW) ============================
    def _bpsdf_h_cb(self, msg: Float32):
        """
        Map barrier value h to haptic intensity.
        h >= h_max  → no vibration (safe)
        h == 0      → full intensity (at boundary)
        h < 0       → full intensity (inside boundary / violated)
        """
        h = float(msg.data)
        if h >= self._h_max:
            self._latest_intensity = 0.0
        else:
            # Linear mapping: intensity = 1 at h<=0, 0 at h>=h_max
            raw_intensity = min(1.0, max(0.0, 1.0 - h / self._h_max))
            self._latest_intensity = self._apply_calibration(raw_intensity)

        self._update_bpsdf_output()

    def _update_bpsdf_output(self):
        """
        Drive haptic output for bpsdf mode.
        - Not flanked: normal continuous L/R blend from /haptic/direction (as before).
        - Flanked (obstacles on both L and R): hand off to the engine's built-in
          corridor cue (sample-accurate L/R ping-pong, same mechanism used for the
          proximity-mode corridor case) instead of writing amplitudes directly.
        """
        if self._flanked:
            peak = self.engine.master * self._latest_intensity
            self.engine.set_corridor(peak, peak)
        else:
            self.engine.clear_corridor()
            left, right = self.engine.directional(self._direction_deg)
            self.engine.ampL_target = self.engine.master * self._latest_intensity * left
            self.engine.ampR_target = self.engine.master * self._latest_intensity * right

    def _angle_from_front(self, bearing_deg):
        """Signed bearing relative to straight ahead (0 deg), wrapped to (-180, 180]."""
        return ((bearing_deg + 180.0) % 360.0) - 180.0

    def _obstacles_cb(self, msg: Float32MultiArray):
        """
        In bpsdf mode, obstacles topic drives two things:
          1. Rear beep gate (unchanged from before).
          2. Flank detection: are there obstacles within flank_max_dist on BOTH
             the left and right side of the front cone? If so, switch the main
             vibration into ping-pong pulsing instead of a static L/R blend.
        """
        data = list(msg.data)
        nearest_behind = None
        has_left = False
        has_right = False
        for i in range(0, len(data) - 1, 2):
            dist, bearing = data[i], data[i + 1]

            if abs(self._angle_from_rear(bearing)) <= self.rear_halfwidth:
                if nearest_behind is None or dist < nearest_behind[0]:
                    nearest_behind = (dist, bearing)

            if dist <= self._flank_max_dist:
                angle = self._angle_from_front(bearing)
                # Points within the front-center dead zone don't count as "left" or
                # "right" — a single obstacle ahead of you often reports points on
                # both sides of 0 deg (its own width), and without this dead zone
                # that gets misread as two obstacles flanking you.
                if angle < -self._flank_deadzone and angle >= -self._flank_cone:
                    has_left = True
                elif angle > self._flank_deadzone and angle <= self._flank_cone:
                    has_right = True

        if nearest_behind is not None:
            self._update_rear_beep(nearest_behind[0], nearest_behind[1])
        else:
            self.engine.set_beep(0.0)

        was_flanked = self._flanked
        self._flanked = has_left and has_right
        if self._flanked != was_flanked:
            self._update_bpsdf_output()

    # ============================ multi-obstacle ============================
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
            if abs(self._angle_from_rear(bearing)) <= self.rear_halfwidth:
                if nearest_behind is None or dist < nearest_behind[0]:
                    nearest_behind = (dist, bearing)

        self.engine.set_multi_targets(obstacles)

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