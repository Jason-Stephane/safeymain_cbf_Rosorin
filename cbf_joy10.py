#!/usr/bin/env python3
# =================================================================================================
# cbf_joy10.py  —  Consolidated CBF safety filter in JOY space with dual kinematic modes.
#
# Self-contained node: no dependency on sac3_apf.py. Integrates LiDAR perception
# directly and feeds dense obstacle points into the CBF.
#
# Dual kinematic model:
#   HOLONOMIC (default):  3D QP (vx, vy, ω) — robot can translate in any direction
#   UNICYCLE (L1 toggle): 2D QP (v, ω)      — robot can only move forward/backward + rotate
#   Toggle L1 button to switch modes at runtime.
#
# Joy axis mapping:
#   Holonomic:   left Y → vx, left X → vy, right X → ω
#   Unicycle:    left Y → v,                right X → ω
#
# =================================================================================================

"""
python3 cbf_joy10.py --ros-args \
  -p cbf_type:=distance_multi_slack_weighted \
  -p barrier_mode:=bpsdf \
  -p sdf_config_path:=/home/developer/Documents/Safety/bp_sdf_coeffs.json
"""


"""
python3 cbf_joy10.py --ros-args \
  -p cbf_type:=distance_multi_slack_weighted \
  -p barrier_mode:=bpsdf \
  -p sdf_config_path:=/home/jasonstephane/Downloads/safety26-main/bp_sdf_coeffs.json
"""
"""python3 cbf_joy10.py --ros-args -p cbf_type:=distance_multi_slack_weighted -p barrier_mode:=bpsdf """

"""python3 cbf_joy10.py --ros-args -p cbf_type:=distance_multi_slack_weighted -p barrier_mode:=bpsdf -p d_safe:=0.5, -p sdf_margin:=0.05"""
import math
import sys

import numpy as np
import tempfile # Added for temporary file generation
# =================================================================================================
# Embedded BPSDF Coefficients
# Paste your actual JSON content inside this string
# =================================================================================================
EMBEDDED_SDF_JSON = """{
  "robot_width": 0.22,
  "robot_length": 0.285,
  "half_w": 0.1425,
  "half_l": 0.11,
  "degree": 14,
  "domain": {
    "x": [
      -0.4425,
      0.4425
    ],
    "y": [
      -0.41,
      0.41
    ]
  },
  "coefficients": [
    0.42004402596281093, 0.4311600734081971, 0.1598073970139166, 0.6595146908141711, 0.11497374136734265,
    -0.3401163372511941, 1.896136246148402, -1.6836286408554788, 1.854223966535948, -0.26126996803453817,
    0.02867865926850736, 0.7165275218901517, 0.13771380489137014, 0.43585911328020205, 0.41966678824242626,
    0.4231513249624448, -0.13333257561268894, 2.0262341235965415, -3.406469311585782, 4.75067665209217,
    -2.7358858770687293, 1.116848705494323, -0.0528465239375862, 1.2417702540429996, -3.089796850011236,
    5.283417364171, -3.8197911102373476, 2.195728307695026, -0.171283163246213, 0.42679825227751084,
    0.21159967434334967, 1.4869763029694432, -3.0026928039149516, 5.3911015976323045, -3.25258596000642,
    -1.9374522510757113, 3.2119996940608866, -0.49855044102544355, 3.671883443579633, -2.3677583620663025,
    -3.6149240859892604, 5.994412736963401, -3.267625115817741, 1.550721163291712, 0.20170707580250313,
    0.42478598112487465, -0.8720734235476423, -0.4956144658495707, 4.711121925381703, -10.782724281113175,
    12.800382798217276, 0.5582537113082537, -12.933420515732672, 0.3879219610200489, 14.522945113802058,
    -11.747430233670482, 4.866060311848049, -0.7007426672262712, -0.8054145019510107, 0.43403482876739613,
    0.5019836777943, 0.702031316202932, 7.294125242697293, -9.693560838204743, 8.683569620075428,
    -1.0820267982350422, -8.427775839027015, 14.485445846113375, -12.122027104246278, -0.9716388364411206,
    9.093405103794419, -10.040147140263107, 8.165282800947848, 0.36445590779584325, 0.506113842601078,
    -0.6610951787256492, -1.2764871555897874, -6.611475667195584, -6.525958159556667, 16.645528177395615,
    -20.72114903937212, -7.845285922727505, 41.86827269220716, -5.018130504686511, -19.35847955849123,
    17.08327033271703, -7.412628832235055, -7.353443450233439, -0.8333744174655456, -0.675827731617867,
    2.0344147198100244, 3.8574815758321357, 4.98422282482271, 12.820164932148503, 1.1411608377169467,
    2.3549300994694717, -10.36299348257614, -21.43262722142562, -8.948884220529994, -1.161971364037981,
    -0.002106037981304447, 15.1157826991602, 4.826801442095041, 3.643081298928015, 2.042370756445597,
    -1.7565314647136354, -4.274858874548807, -4.818022508368827, -7.082373651170069, -32.24388462971476,
    30.792297862504096, 26.266300029929734, -20.53926838255905, 26.26629996054626, 30.792297626812996,
    -32.24388460131489, -7.082373589749249, -4.818022510037091, -4.27485888922274, -1.7565314606594846,
    2.0423707587794797, 3.6430812823798173, 4.826801471472801, 15.1157827307775, -0.0021060757905184005,
    -1.161971533519857, -8.94888419361352, -21.432627380007396, -10.362993221381254, 2.3549302099783485,
    1.141160850785087, 12.820164796871873, 4.98422288030804, 3.8574815808410947, 2.0344147151147136,
    -0.6758277319927339, -0.8333744125719044, -7.353443444660671, -7.412628957604718, 17.083270590237888,
    -19.358479685347653, -5.018130391537793, 41.86827275636213, -7.845286078300376, -20.721149240767822,
    16.645528315821096, -6.525958110387826, -6.611475709081984, -1.2764871593811031, -0.6610951744283255,
    0.5061138421808343, 0.3644559100403268, 8.165282778200348, -10.040147017705307, 9.093404874258587,
    -0.9716387000518615, -12.122027180600055, 14.485445833313271, -8.427775835804763, -1.082026518118284,
    8.683569369807241, -9.693560783682505, 7.294125247869111, 0.7020313212269569, 0.5019836749154803,
    0.43403482892147166, -0.8054145016385735, -0.7007426663083223, 4.86606028999536, -11.747430196145451,
    14.522945127527596, 0.3879219448601889, -12.933420505999296, 0.5582537230839754, 12.800382667889684,
    -10.78272416751582, 4.711121906238593, -0.49561447428463007, -0.8720734236932108, 0.42478598221927044,
    0.20170707590588768, 1.550721161038866, -3.2676251054736225, 5.994412715636946, -3.6149240584802573,
    -2.367758383916586, 3.6718834252439985, -0.4985504086748846, 3.2119996977352976, -1.9374522710798794,
    -3.2525859261246537, 5.391101560078982, -3.0026927845755846, 1.4869762994365892, 0.21159967427257576,
    0.42679825220557993, -0.1712831621475766, 2.1957283028614425, -3.8197911004456695, 5.2834173566331994,
    -3.0897968601314987, 1.2417702905599708, -0.05284656185234861, 1.116848710103318, -2.7358858502119556,
    4.75067661614329, -3.406469285863063, 2.02623411306427, -0.13333257354898032, 0.42315132487651985,
    0.4196667882523443, 0.4358591131531553, 0.13771380535501995, 0.7165275214183059, 0.02867865765197964,
    -0.26126996104491373, 1.8542239536263894, -1.6836286276178412, 1.8961362396596426, -0.3401163384202265,
    0.11497374577768478, 0.6595146873678548, 0.15980739842126118, 0.43116007312252047, 0.42004402597907387
  ],
  "n_coefficients": 225
}"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener
import tf_transformations

from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Header, Float32, Float32MultiArray, ColorRGBA, String
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan, PointCloud2, Joy
from sensor_msgs_py import point_cloud2

from cbf_core7 import (
    CBFParams,
    CBFResult,
    cbf_filter_multi,
    cbf_filter_multi_slack,
    cbf_filter_multi_slack_weighted,
    format_result,
)

# =================================================================================================
# Valid CBF types
# =================================================================================================
VALID_TYPES = (
    'distance_multi',
    'distance_multi_slack',
    'distance_multi_slack_weighted',
)


class CBFJoyFilter(Node):

    def __init__(self):
        super().__init__('cbf_joy_filter_node')
        # =====================================================================
        # Generate Temporary JSON File for CBFParams
        # =====================================================================
        self.temp_sdf_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_sdf_file.write(EMBEDDED_SDF_JSON)
        self.temp_sdf_file.flush()



        # =====================================================================
        # Parameters
        # =====================================================================
        self.declare_parameter('base_frame', 'base_link')
        self.base_frame = self.get_parameter('base_frame').value

        # ---- Topics ----
        self.declare_parameter('scan_topic', '/scan_raw')
        self.declare_parameter('joy_topic', '/ps5/joy')
        self.declare_parameter('output_cmd_topic', '/controller/cmd_vel')

        self.scan_topic       = str(self.get_parameter('scan_topic').value)
        self.joy_topic        = str(self.get_parameter('joy_topic').value)
        self.output_cmd_topic = str(self.get_parameter('output_cmd_topic').value)

        # ---- Joy axis mapping ----
        self.declare_parameter('axis_vx', 1)          # left stick Y → forward/backward
        self.declare_parameter('axis_vy', 0)          # left stick X → lateral (holonomic only)
        self.declare_parameter('axis_wz', 2)          # right stick X → rotation
        self.axis_vx = int(self.get_parameter('axis_vx').value)
        self.axis_vy = int(self.get_parameter('axis_vy').value)
        self.axis_wz = int(self.get_parameter('axis_wz').value)

        # ---- Velocity scaling ----
        self.declare_parameter('max_linear_speed', 0.3)   # m/s at full stick
        self.declare_parameter('max_angular_speed', 1.0)  # rad/s at full stick
        self.max_linear  = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)

        # ---- Buttons ----
        self.declare_parameter('deadman_button', 10)      # R1 — must hold for motion
        self.declare_parameter('mode_button', 9)          # L1 — toggle holonomic/unicycle
        self.deadman_button = int(self.get_parameter('deadman_button').value)
        self.mode_button    = int(self.get_parameter('mode_button').value)

        # ---- CBF type selection ----
        self.declare_parameter('cbf_type', 'distance_multi_slack_weighted')
        self.cbf_type = str(self.get_parameter('cbf_type').value)
        if self.cbf_type not in VALID_TYPES:
            raise ValueError(f"cbf_type must be one of {VALID_TYPES}; got '{self.cbf_type}'")

        # ---- Perception ----
        self.declare_parameter('influence_radius', 0.5)
        self.declare_parameter('scan_max_range', 5.0)
        self.declare_parameter('loop_frequency', 250.0)

        self.influence_radius = float(self.get_parameter('influence_radius').value)
        self.scan_max_range   = float(self.get_parameter('scan_max_range').value)
        self.loop_frequency   = float(self.get_parameter('loop_frequency').value)

        # ---- CBF tuning ----
        self.declare_parameter('barrier_mode', 'bpsdf')
        self.declare_parameter('sdf_margin', 0.025)
        #self.declare_parameter('sdf_config_path', '')
        self.declare_parameter('alpha', 0.75)
        self.declare_parameter('d_safe', 0.25)
        self.declare_parameter('p_slack', 15.0)
        self.declare_parameter('p_base', 0.8)
        self.declare_parameter('eta_w', 0.5)
        self.declare_parameter('w_weight', 0.1)
        self.declare_parameter('default_kinematic_model', 'holonomic')

        self.kinematic_model = str(self.get_parameter('default_kinematic_model').value)

        self.params = CBFParams(
            barrier_mode=str(self.get_parameter('barrier_mode').value),
            sdf_margin=float(self.get_parameter('sdf_margin').value),

            sdf_config_path=self.temp_sdf_file.name,

            alpha=float(self.get_parameter('alpha').value),
            d_safe=float(self.get_parameter('d_safe').value),
            p_slack=float(self.get_parameter('p_slack').value),
            p_base=float(self.get_parameter('p_base').value),
            eta_w=float(self.get_parameter('eta_w').value),
            w_weight=float(self.get_parameter('w_weight').value),
            kinematic_model=self.kinematic_model,
        )

        # ---- Haptic ----
        self.declare_parameter('haptic_bearing_sign', -1.0)
        self.haptic_bearing_sign = float(self.get_parameter('haptic_bearing_sign').value)

        # ---- Robot dimensions for boundary marker ----
        self.declare_parameter('robot_half_x', 0.1425)
        self.declare_parameter('robot_half_y', 0.11)
        self.robot_half_x = float(self.get_parameter('robot_half_x').value)
        self.robot_half_y = float(self.get_parameter('robot_half_y').value)

        # =====================================================================
        # State
        # =====================================================================
        self.front_minAngle     = 0.0
        self.front_maxAngle     = 0.0
        self.front_angIncrement = 0.0
        self.front_minRange     = 0.0
        self.front_maxRange     = 0.0
        self.front_ranges       = []

        self.obstacle_points    = []
        self.base_r_values      = []
        self.base_theta_values  = []
        self.latest_h           = 1.0

        self.deadman_held       = False
        self._prev_mode_btn     = 0       # for edge-detect toggle
        self.display_specs_once = False

        # =====================================================================
        # TF2
        # =====================================================================
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # =====================================================================
        # Dispatch table
        # =====================================================================
        self._dispatch = {
            'distance_multi':                 cbf_filter_multi,
            'distance_multi_slack':           cbf_filter_multi_slack,
            'distance_multi_slack_weighted':  cbf_filter_multi_slack_weighted,
        }

        # =====================================================================
        # Publishers
        # =====================================================================
        self.cmd_pub          = self.create_publisher(Twist,             self.output_cmd_topic, 10)
        self.h_pub            = self.create_publisher(Float32,           '/cbf/h', 10)
        self.mode_pub         = self.create_publisher(String,            '/cbf/kinematic_mode', 10)
        self.desired_pub      = self.create_publisher(Marker,            '/cbf/desired_marker', 10)
        self.safe_pub         = self.create_publisher(Marker,            '/cbf/safe_marker', 10)
        self.boundary_pub     = self.create_publisher(Marker,            '/cbf/boundary_marker', 10)
        self.obstacle_viz_pub = self.create_publisher(Marker,            '/obstacle_marker', 10)
        self.pcld2_pub        = self.create_publisher(PointCloud2,       '/transformed_pcld2', 10)

        # Haptic publishers
        self.haptic_pub       = self.create_publisher(Point,             '/haptic/obstacle', 10)
        self.haptic_dir_pub   = self.create_publisher(Float32,           '/haptic/direction', 10)
        self.haptic_multi_pub = self.create_publisher(Float32MultiArray, '/haptic/obstacles', 10)
        self.haptic_delta_pub = self.create_publisher(Float32,           '/haptic/cbf_delta', 10)

        # =====================================================================
        # Subscribers
        # =====================================================================
        self.create_subscription(LaserScan, self.scan_topic,  self._scan_cb, 10)
        self.create_subscription(Joy,       self.joy_topic,   self._joy_cb, 10)

        # =====================================================================
        # Main perception loop
        # =====================================================================
        self.create_timer(1.0 / self.loop_frequency, self._main_loop)

        self.get_logger().info(
            f'CBF Joy Filter (consolidated, dual-kinematic) active.\n'
            f'  cbf_type       : {self.cbf_type}\n'
            f'  barrier_mode   : {self.params.barrier_mode}\n'
            f'  default_model  : {self.kinematic_model}\n'
            f'  influence_radius: {self.influence_radius}\n'
            f'  deadman_button : {self.deadman_button} (hold)\n'
            f'  mode_button    : {self.mode_button} (toggle holo/uni)\n'
            f'  JOY IN : {self.joy_topic}\n'
            f'  CMD OUT: {self.output_cmd_topic}\n'
            f'  SCAN   : {self.scan_topic}'
        )

    # =====================================================================
    # Callbacks
    # =====================================================================

    def _scan_cb(self, msg: LaserScan):
        """Cache latest LiDAR scan."""
        self.front_minAngle     = msg.angle_min
        self.front_maxAngle     = msg.angle_max
        self.front_angIncrement = msg.angle_increment
        self.front_minRange     = msg.range_min
        self.front_maxRange     = msg.range_max
        self.front_ranges       = list(msg.ranges)

    def _joy_cb(self, msg: Joy):
        """
        Main control callback: read joystick, toggle mode, map axes, filter through CBF, publish.
        """
        # ---- Deadman switch ----
        self.deadman_held = (msg.buttons[self.deadman_button] == 1
                             if self.deadman_button < len(msg.buttons) else False)

        # ---- Mode toggle (edge-detect: trigger on press, not hold) ----
        curr_mode_btn = (msg.buttons[self.mode_button]
                         if self.mode_button < len(msg.buttons) else 0)
        if curr_mode_btn == 1 and self._prev_mode_btn == 0:
            # Toggle
            if self.kinematic_model == 'holonomic':
                self.kinematic_model = 'unicycle'
            else:
                self.kinematic_model = 'holonomic'
            self.params.kinematic_model = self.kinematic_model
            self.get_logger().info(f'Kinematic mode switched to: {self.kinematic_model}')
        self._prev_mode_btn = curr_mode_btn

        # Publish current mode for monitoring
        self.mode_pub.publish(String(data=self.kinematic_model))

        # ---- Map joy axes to desired velocity ----
        raw_vx = msg.axes[self.axis_vx] if self.axis_vx < len(msg.axes) else 0.0
        raw_vy = msg.axes[self.axis_vy] if self.axis_vy < len(msg.axes) else 0.0
        raw_wz = msg.axes[self.axis_wz] if self.axis_wz < len(msg.axes) else 0.0

        vdes_x = raw_vx * self.max_linear
        vdes_wz = raw_wz * self.max_angular

        if self.kinematic_model == 'holonomic':
            vdes_y = raw_vy * self.max_linear
        else:
            vdes_y = 0.0    # unicycle cannot strafe

        # ---- Passthrough if deadman not held ----
        if not self.deadman_held:
            self.cmd_pub.publish(Twist())
            # out = Twist()
            # out.linear.x = vdes_x
            # out.linear.y = vdes_y
            # out.angular.z = vdes_wz
            # self.cmd_pub.publish(out)
            return

        # ---- CBF filter ----
        obstacles = [(p.x, p.y) for p in self.obstacle_points]

        try:
            fn = self._dispatch[self.cbf_type]
            result = fn(vdes_x, vdes_y, obstacles, self.params, vdes_wz=vdes_wz)
        except Exception as e:
            self.get_logger().error(f'CBF filter crashed: {e}', throttle_duration_sec=1.0)
            out = Twist()
            out.linear.x, out.linear.y, out.angular.z = vdes_x, vdes_y, vdes_wz
            self.cmd_pub.publish(Twist())
            return

        # ---- Publish safe command ----
        out = Twist()
        out.linear.x  = result.vx
        out.linear.y  = result.vy
        out.angular.z = result.wz
        self.cmd_pub.publish(out)

        # ---- Update h for visualization ----
        self.latest_h = result.h if result.h is not None else 1.0
        self.h_pub.publish(Float32(data=float(self.latest_h)))

        # ---- Haptic delta ----
        u_des_norm  = math.hypot(vdes_x, vdes_y)
        u_safe_norm = math.hypot(result.vx, result.vy)
        self.haptic_delta_pub.publish(Float32(data=abs(u_des_norm - u_safe_norm)))

        # ---- Visualization arrows ----
        self._publish_desired_arrow(vdes_x, vdes_y)
        self._publish_safe_arrow(result.vx, result.vy)

        self.get_logger().info(
            f'[{self.kinematic_model[:4]}] ' + format_result(result),
            throttle_duration_sec=1.0
        )

    # =====================================================================
    # Perception pipeline
    # =====================================================================

    def _main_loop(self):
        """Perception cycle: transform scan, extract obstacles, publish viz."""
        if not self.front_ranges:
            return

        if not self.display_specs_once:
            self._log_laser_specs()
            self.display_specs_once = True

        # Transform lidar → base_link and collect dense obstacle points
        self._transform_and_detect()

        # Publish visualizations
        self._publish_obstacle_points()
        self._publish_boundary_marker()
        self._publish_haptic()

    def _transform_and_detect(self):
        """
        Transform all valid LiDAR points into base_link frame.
        Collect every point within influence_radius as an obstacle constraint.
        """
        trans, rot = self._get_transform(self.base_frame, 'lidar_frame')
        if trans is None:
            return

        transformed_points = []
        self.obstacle_points = []
        self.base_r_values = []
        self.base_theta_values = []

        # Build the 4x4 transform once (not per-point)
        Rq = tf_transformations.quaternion_matrix(rot)
        Tr = np.array([
            [1, 0, 0, trans[0]],
            [0, 1, 0, trans[1]],
            [0, 0, 1, trans[2]],
            [0, 0, 0, 1       ],
        ])
        Tx = np.dot(Tr, Rq)

        for i, r in enumerate(self.front_ranges):
            if not (self.front_minRange < r < min(self.front_maxRange, self.scan_max_range)):
                continue

            theta = self.front_minAngle + i * self.front_angIncrement
            x_lidar = r * math.cos(theta)
            y_lidar = r * math.sin(theta)
            lidar_pt = np.array([x_lidar, y_lidar, 0.0, 1.0])
            base_pt = np.dot(Tx, lidar_pt)

            x_b = float(base_pt[0])
            y_b = float(base_pt[1])
            z_b = float(base_pt[2])
            transformed_points.append((x_b, y_b, z_b))

            dist = math.hypot(x_b, y_b)
            self.base_r_values.append(dist)
            self.base_theta_values.append(math.atan2(y_b, x_b))

            # Dense obstacle sampling: every point within influence radius feeds the QP
            if dist <= self.influence_radius:
                self.obstacle_points.append(Point(x=x_b, y=y_b, z=0.0))

        # Publish full point cloud
        if transformed_points:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.base_frame
            pcld2_msg = point_cloud2.create_cloud_xyz32(header, transformed_points)
            self.pcld2_pub.publish(pcld2_msg)

    def _get_transform(self, target, source):
        """Look up TF2 transform. Returns (translation, rotation) or (None, None)."""
        try:
            now = rclpy.time.Time()
            t = self.tf_buffer.lookup_transform(target, source, now, timeout=Duration(seconds=0.5))
            trans = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
            rot = (t.transform.rotation.x, t.transform.rotation.y,
                   t.transform.rotation.z, t.transform.rotation.w)
            return trans, rot
        except Exception as e:
            self.get_logger().error(f'TF {source}→{target} failed: {e}', throttle_duration_sec=2.0)
            return None, None

    def _log_laser_specs(self):
        """Log LiDAR specs once at startup."""
        fov = self.front_maxAngle - self.front_minAngle
        n = int(math.degrees(fov) / math.degrees(self.front_angIncrement)) if self.front_angIncrement else 0
        self.get_logger().info(
            f'LiDAR: FoV={math.degrees(fov):.1f}°  beams={n}  '
            f'range=[{self.front_minRange:.2f}, {self.front_maxRange:.2f}]m  '
            f'increment={math.degrees(self.front_angIncrement):.3f}°'
        )

    # =====================================================================
    # Visualization
    # =====================================================================

    def _publish_desired_arrow(self, vx, vy):
        """Blue arrow: operator's raw commanded velocity."""
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'cbf_desired'
        m.type = Marker.ARROW
        m.scale.x, m.scale.y, m.scale.z = 0.04, 0.08, 0.1
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.4, 1.0, 0.9
        m.pose.orientation.w = 1.0
        m.points = [Point(x=0.0, y=0.0, z=0.0), Point(x=float(vx), y=float(vy), z=0.0)]
        m.lifetime = Duration(seconds=0.2).to_msg()
        self.desired_pub.publish(m)

    def _publish_safe_arrow(self, vx, vy):
        """Green arrow: CBF-filtered safe velocity."""
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'cbf_safe'
        m.type = Marker.ARROW
        m.scale.x, m.scale.y, m.scale.z = 0.05, 0.1, 0.12
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.3, 1.0
        m.pose.orientation.w = 1.0
        m.points = [Point(x=0.0, y=0.0, z=0.0), Point(x=float(vx), y=float(vy), z=0.0)]
        m.lifetime = Duration(seconds=0.2).to_msg()
        self.safe_pub.publish(m)

    def _publish_obstacle_points(self):
        """Red dots: all obstacle points currently feeding the QP."""
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'obstacles'
        m.type = Marker.POINTS
        m.scale.x, m.scale.y = 0.025, 0.025
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.1, 0.8
        m.pose.orientation.w = 1.0
        m.points = self.obstacle_points
        m.lifetime = Duration(seconds=0.15).to_msg()
        self.obstacle_viz_pub.publish(m)

    def _publish_boundary_marker(self):
        """
        Rectangle outline around robot that changes color based on barrier value h.
        Green (safe) → Yellow (caution) → Red (danger).
        """
        h = self.latest_h

        # Color gradient based on h
        # h > 0.10 → green,  h ~ 0.05 → yellow,  h < 0.02 → red
        if h > 0.10:
            r, g, b = 0.1, 0.9, 0.2
        elif h > 0.02:
            # Linear interpolation green → yellow → red
            t = max(0.0, min(1.0, (h - 0.02) / 0.08))   # 1 at h=0.10, 0 at h=0.02
            r = 1.0 - 0.9 * t    # 0.1 → 1.0
            g = 0.2 + 0.7 * t    # 0.9 → 0.2
            b = 0.0
        else:
            r, g, b = 1.0, 0.0, 0.0

        hx = self.robot_half_x
        hy = self.robot_half_y
        corners = [
            Point(x= hx, y= hy, z=0.01),
            Point(x= hx, y=-hy, z=0.01),
            Point(x=-hx, y=-hy, z=0.01),
            Point(x=-hx, y= hy, z=0.01),
            Point(x= hx, y= hy, z=0.01),  # close the rectangle
        ]

        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'safety_boundary'
        m.type = Marker.LINE_STRIP
        m.scale.x = 0.02      # line width
        m.color.r, m.color.g, m.color.b = r, g, b
        m.color.a = 0.9
        m.pose.orientation.w = 1.0
        m.points = corners
        m.lifetime = Duration(seconds=0.15).to_msg()
        self.boundary_pub.publish(m)

    # =====================================================================
    # Haptic feedback
    # =====================================================================

    def _publish_haptic(self):
        """Publish haptic data: both single-obstacle and multi-obstacle formats."""
        # Find nearest obstacle
        if not self.obstacle_points:
            self.haptic_pub.publish(Point(x=99.0, y=0.0, z=0.0))
            self.haptic_dir_pub.publish(Float32(data=0.0))
            self.haptic_multi_pub.publish(Float32MultiArray(data=[99.0, 0.0]))
            return

        nearest = min(self.obstacle_points, key=lambda p: math.hypot(p.x, p.y))
        dist_m = math.hypot(nearest.x, nearest.y)
        bearing_deg = self.haptic_bearing_sign * math.degrees(math.atan2(nearest.y, nearest.x))

        # Single-obstacle (backward compat)
        self.haptic_pub.publish(Point(x=float(dist_m), y=float(bearing_deg), z=0.0))
        self.haptic_dir_pub.publish(Float32(data=float(bearing_deg)))

        # Multi-obstacle
        pairs = []
        for p in self.obstacle_points:
            d = math.hypot(p.x, p.y)
            if d < self.influence_radius:
                b = self.haptic_bearing_sign * math.degrees(math.atan2(p.y, p.x))
                pairs.extend([d, b])
        self.haptic_multi_pub.publish(Float32MultiArray(data=pairs if pairs else [99.0, 0.0]))

    def destroy_node(self):
        """Clean up the temporary JSON file before shutting down."""
        import os
        try:
            os.remove(self.temp_sdf_file.name)
        except OSError:
            pass
        super().destroy_node()


# =================================================================================================
# Entry point
# =================================================================================================
def main(args=None):
    rclpy.init(args=sys.argv)
    node = CBFJoyFilter()
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
