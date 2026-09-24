#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# ROS 2 Humble port of sharedAutonomyController.py
#
# This node implements a shared autonomy controller for a mobile robot.
# It blends human joystick input with repulsive potential fields derived
# from LiDAR scan data to provide obstacle avoidance assistance.
#
# UPDATED: Directly publishes processed Joy messages to /ps5_control_node
# -------------------------------------------------------------------------------------------------

import math
import numpy as np
import sys
from scipy.spatial import ConvexHull

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener
import tf_transformations

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan, PointCloud2, Joy
from sensor_msgs_py import point_cloud2

# -------------------------------------------------------------------------------------------------
# Global constants
# -------------------------------------------------------------------------------------------------

# Repulsive force computation mode:
#   ALL_OBSTACLES  — sum repulsion vectors from every detected obstacle cluster
#   CLOSEST_OBSTACLE — use only the strongest (closest) repulsion vector
ALL_OBSTACLES = 0
CLOSEST_OBSTACLE = 1

# Normalization divisors applied to repulsion, resultant, and final velocity vectors.
NORM_FACTOR_0 = 1   # Per-obstacle repulsion vector
NORM_FACTOR_1 = 1   # Repulsive resultant vector
NORM_FACTOR_2 = 1   # Final blended velocity signal

# Repulsive potential field gain constant.
K_REP = 0.375


class SharedAutonomyController(Node):
    def __init__(self):
        super().__init__('shared_autonomy_controller_node')

        self.declare_parameter('base_frame', 'base_link') 
        self.base_frame = self.get_parameter('base_frame').value 

        # ----------------------------------------------------------------------------
        # ROS 2 parameters — can be overridden via YAML config or CLI arguments
        # ----------------------------------------------------------------------------
        self.declare_parameter('scan_topic', '/scan_raw')
        self.declare_parameter('joy_topic', '/ps5/joy')
        self.declare_parameter('output_joy_topic', '/ps5_control_node')
        self.declare_parameter('loop_frequency', 250.0)         # Hz — main control loop rate
        self.declare_parameter('marker_threshold_range', 2.0)   # m — max range to visualize obstacles
        self.declare_parameter('scan_threshold_range', 5.0)     # m — max range to include scan points
        self.declare_parameter('rho_0', 0.5)    # m — influence radius; repulsion is zero beyond this
        self.declare_parameter('rho_cap', 0.3)  # m — saturation radius; repulsion is clamped inside this
        self.declare_parameter('rep_from', CLOSEST_OBSTACLE)    # Which obstacles contribute to repulsion

        self.loop_frequency         = float(self.get_parameter('loop_frequency').value)
        self.marker_threshold_range = float(self.get_parameter('marker_threshold_range').value)
        self.scan_threshold_range   = float(self.get_parameter('scan_threshold_range').value)
        self.rho_0                  = float(self.get_parameter('rho_0').value)
        self.rho_cap                = float(self.get_parameter('rho_cap').value)
        self.rep_from               = int(self.get_parameter('rep_from').value)

        # Marker lifetime is tied to the loop period so stale markers auto-expire
        self.marker_lifetime = 1.0 / self.loop_frequency
        self.k_rep = K_REP

        # ----------------------------------------------------------------------------
        # LiDAR scan metadata — populated on first scan message
        # ----------------------------------------------------------------------------
        self.front_minAngle     = 0.0
        self.front_maxAngle     = 0.0
        self.front_angIncrement = 0.0
        self.front_minRange     = 0.0
        self.front_maxRange     = 0.0
        self.front_ranges       = []
        self.front_FoV          = 0.0   
        self.front_noOfScans    = 0.0   

        # ----------------------------------------------------------------------------
        # Working data — reset every control loop iteration
        # ----------------------------------------------------------------------------
        self.base_r_values      = []   
        self.base_theta_values  = []   
        self.roi1_ranges        = []   
        self.centroids          = []   
        self.closestPoints      = []   
        self.rep_points         = []   

        # ----------------------------------------------------------------------------
        # Velocity vectors used in the blending pipeline
        # ----------------------------------------------------------------------------
        self.rep_resultant  = Point(x=0.0, y=0.0, z=0.0)  
        self.ref_signal     = Point(x=0.0, y=0.0, z=0.0)  
        self.vfinal_signal  = Point(x=0.0, y=0.0, z=0.0)  

        # ----------------------------------------------------------------------------
        # Joystick / operator state
        # ----------------------------------------------------------------------------
        self.deadman_switch  = 0        # Initialized to 0 (unpressed) for safety cut-off
        self.autonomy_switch = 1        
        self.vfinal_joy      = Joy()    

        # ----------------------------------------------------------------------------
        # TF2 — used to transform LiDAR points from lidar_frame into base_link
        # ----------------------------------------------------------------------------
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----------------------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------------------
        self.pcld2_pub       = self.create_publisher(PointCloud2,  '/transformed_pcld2', 10)
        self.marker_pub      = self.create_publisher(Marker,       '/obstacle_marker',   10)
        self.centroid_pub    = self.create_publisher(Marker,       '/centroid_marker',   10)
        self.scan1_pub       = self.create_publisher(LaserScan,    '/ROI1_laserScan',    10)
        self.repForce_pub    = self.create_publisher(MarkerArray,  '/rep_marker',        10)
        self.resForce_pub    = self.create_publisher(Marker,       '/res_marker',        10)
        self.refSignal_pub   = self.create_publisher(Marker,       '/refSignal_marker',  10)
        self.vfinal_marker_pub = self.create_publisher(Marker,     '/vfinal_marker',     10)

        output_joy_topic = self.get_parameter('output_joy_topic').value
        self.vfinal_joy_pub = self.create_publisher(Joy, output_joy_topic, 10)

        # ----------------------------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------------------------
        scan_topic = self.get_parameter('scan_topic').value
        joy_topic  = self.get_parameter('joy_topic').value

        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.frontScan_callback, 10
        )
        self.joy_sub = self.create_subscription(
            Joy, joy_topic, self.joy_bs_callback, 10
        )

        # ----------------------------------------------------------------------------
        # Main control loop timer
        # ----------------------------------------------------------------------------
        self.main_timer = self.create_timer(1.0 / self.loop_frequency, self.main_loop)

        self.displayLaserSpecs_once = False  
        
        self.get_logger().info(f'Using base_frame = {self.base_frame}') 
    

    # =============================================================================
    # Callbacks
    # =============================================================================

    def frontScan_callback(self, scan_msg: LaserScan):
        """Cache the latest LiDAR scan metadata and range array."""
        self.front_minAngle     = scan_msg.angle_min
        self.front_maxAngle     = scan_msg.angle_max
        self.front_angIncrement = scan_msg.angle_increment
        self.front_FoV          = self.front_maxAngle - self.front_minAngle

        if self.front_angIncrement != 0.0:
            self.front_noOfScans = math.degrees(self.front_FoV) / math.degrees(self.front_angIncrement)
        else:
            self.front_noOfScans = 0.0

        self.front_minRange = scan_msg.range_min
        self.front_maxRange = scan_msg.range_max
        self.front_ranges   = list(scan_msg.ranges)

    def joy_bs_callback(self, joy_bs_msg: Joy):
        """
        Cache the latest joystick message and extract the operator's velocity intent.
        """
        self.vfinal_joy = joy_bs_msg  

        if len(joy_bs_msg.axes) >= 2:
            self.ref_signal.x = joy_bs_msg.axes[1]
            self.ref_signal.y = joy_bs_msg.axes[0]
        else:
            self.ref_signal.x = 0.0
            self.ref_signal.y = 0.0
        self.ref_signal.z = 0.0

        # Read L2 switch status (index 6). If missing, defaults to 0 (safe).
        self.deadman_switch = joy_bs_msg.buttons[6] if len(joy_bs_msg.buttons) > 6 else 0

    # =============================================================================
    # Helpers
    # =============================================================================

    def getAngle(self, range_index: int) -> float:
        return self.front_minAngle + (range_index * self.front_angIncrement)

    def getTransform(self, target_frame: str, source_frame: str):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, now, timeout=Duration(seconds=0.5)
            )
            trans = (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            )
            rot = (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            )
            return trans, rot
        except Exception as e:
            self.get_logger().error(
                f'Transform lookup from {source_frame} to {target_frame} failed: {e}'
            )
            return None, None

    def publish_transformed_pointCloud(self):
        trans, rot = self.getTransform(self.base_frame, 'lidar_frame')

        if trans is None or rot is None:
            self.get_logger().warn('Frame transformation skipped.')
            return

        transformed_points    = []
        self.base_r_values    = []
        self.base_theta_values = []

        for i, r in enumerate(self.front_ranges):
            if self.front_minRange < r < self.front_maxRange:
                theta = self.getAngle(i)

                x_lidar = r * math.cos(theta)
                y_lidar = r * math.sin(theta)
                z_lidar = 0.0
                lidar_point = np.array([x_lidar, y_lidar, z_lidar, 1.0])  

                Rq = tf_transformations.quaternion_matrix(rot)
                Tr = np.array([
                    [1, 0, 0, trans[0]],
                    [0, 1, 0, trans[1]],
                    [0, 0, 1, trans[2]],
                    [0, 0, 0, 1       ],
                ])
                Tx = np.dot(Tr, Rq)

                base_point = np.dot(Tx, lidar_point)
                x_base = float(base_point[0])
                y_base = float(base_point[1])
                z_base = float(base_point[2])
                transformed_points.append((x_base, y_base, z_base))

                r_base     = math.sqrt(x_base**2 + y_base**2)
                theta_base = math.atan2(y_base, x_base)
                self.base_r_values.append(r_base)
                self.base_theta_values.append(theta_base)

        header           = Header()
        header.stamp     = self.get_clock().now().to_msg()
        header.frame_id  = 'base_link'
        pcld2_msg = point_cloud2.create_cloud_xyz32(header, transformed_points)
        self.pcld2_pub.publish(pcld2_msg)

    def displayLaserSpecs(self):
        self.get_logger().info('*** ROSOrin+ Front Laser Specifications ***')
        self.get_logger().info(f'Minimum Angle in degrees: {math.degrees(self.front_minAngle)}')
        self.get_logger().info(f'Maximum Angle in degrees: {math.degrees(self.front_maxAngle)}')
        self.get_logger().info(f'Angle increment in degrees: {math.degrees(self.front_angIncrement)}')
        self.get_logger().info(f'FoV in degrees: {math.degrees(self.front_FoV)}')
        self.get_logger().info(f'Number of scans per sweep: {self.front_noOfScans}')
        self.get_logger().info(f'Min Range scanned in m: {self.front_minRange}')
        self.get_logger().info(f'Max Range scanned in m: {self.front_maxRange}')
        self.get_logger().info(f'Range array size: {len(self.front_ranges)}')
        self.get_logger().info('*** *** ***')

    def record_closestPoint(self, points):
        cPoint = Point()
        min_distance = 5.5  
        for point in points:
            distance = math.sqrt(point.x**2 + point.y**2)
            if distance <= min_distance:
                min_distance = distance
                cPoint = point
        self.closestPoints.append(cPoint)

    def compute_centroid(self, points):
        n = len(points)
        if n == 0:
            return Point(x=0.0, y=0.0, z=0.0)
        g_x = sum(p.x for p in points) / n
        g_y = sum(p.y for p in points) / n
        centroid = Point(x=g_x, y=g_y, z=0.0)
        self.centroids.append(centroid)
        return centroid

    def publish_centroids(self):
        centroid_marker = Marker()
        centroid_marker.header.frame_id  = 'base_link'
        centroid_marker.header.stamp     = self.get_clock().now().to_msg()
        centroid_marker.type             = Marker.POINTS
        centroid_marker.scale.x          = 0.05
        centroid_marker.scale.y          = 0.05
        centroid_marker.color.r          = 0.0
        centroid_marker.color.g          = 1.0
        centroid_marker.color.b          = 0.0
        centroid_marker.color.a          = 1.0
        centroid_marker.points           = self.centroids
        centroid_marker.pose.orientation.w = 1.0
        centroid_marker.lifetime         = Duration(seconds=self.marker_lifetime).to_msg()
        self.centroid_pub.publish(centroid_marker)

    def compute_convexhull(self, points):
        if len(points) < 3:
            return points
        points_np = np.array([(p.x, p.y) for p in points])
        cvxhull   = ConvexHull(points_np)
        return [Point(x=p[0], y=p[1], z=0.0) for p in points_np[cvxhull.vertices]]

    def compute_roi1(self, points, centroid):
        for point in points:
            dir_x  = point.x - centroid.x
            dir_y  = point.y - centroid.y
            length = math.sqrt(dir_x**2 + dir_y**2)
            if length == 0.0:
                continue

            unit_x = dir_x / length
            unit_y = dir_y / length

            roi1_x = point.x + unit_x * self.rho_0
            roi1_y = point.y + unit_y * self.rho_0
            roi1_r     = math.sqrt(roi1_x**2 + roi1_y**2)
            roi1_theta = math.atan2(roi1_y, roi1_x)

            if self.front_minAngle <= roi1_theta <= self.front_maxAngle:
                index = int((roi1_theta - self.front_minAngle) / self.front_angIncrement)
                if 0 <= index < len(self.roi1_ranges):
                    self.roi1_ranges[index] = roi1_r

    def publish_roi1(self):
        scan1                  = LaserScan()
        scan1.header.frame_id  = 'base_link'
        scan1.header.stamp     = self.get_clock().now().to_msg()
        scan1.angle_min        = self.front_minAngle
        scan1.angle_max        = self.front_maxAngle
        scan1.angle_increment  = self.front_angIncrement
        scan1.range_min        = self.front_minRange
        scan1.range_max        = self.front_maxRange
        scan1.ranges           = self.roi1_ranges
        self.scan1_pub.publish(scan1)

    def publish_potentialFields(self):
        rep_markerArray = MarkerArray()
        self.rep_points = []
        self.rho_values = []

        for i, cPoint in enumerate(self.closestPoints):
            rho_xy = math.sqrt(cPoint.x**2 + cPoint.y**2)
            self.rho_values.append(rho_xy)
            if rho_xy == 0.0:
                continue  

            rep_unit_x = -cPoint.x / rho_xy
            rep_unit_y = -cPoint.y / rho_xy

            if self.rho_cap <= rho_xy <= self.rho_0:
                scale  = (self.k_rep / rho_xy**2) * ((1.0 / rho_xy) - (1.0 / self.rho_0))
                rep_x  = rep_unit_x * scale / NORM_FACTOR_0
                rep_y  = rep_unit_y * scale / NORM_FACTOR_0
            elif rho_xy < self.rho_cap:
                scale  = (self.k_rep / self.rho_cap**2) * ((1.0 / self.rho_cap) - (1.0 / self.rho_0))
                rep_x  = rep_unit_x * scale / NORM_FACTOR_0
                rep_y  = rep_unit_y * scale / NORM_FACTOR_0
            else:
                rep_x, rep_y = 0.0, 0.0

            rep_point = Point(x=rep_x, y=rep_y, z=0.0)
            self.rep_points.append(rep_point)

            rep_marker                      = Marker()
            rep_marker.header.frame_id      = 'base_link'
            rep_marker.header.stamp         = self.get_clock().now().to_msg()
            rep_marker.type                 = Marker.ARROW
            rep_marker.id                   = i
            rep_marker.ns                   = 'repulsive_forces'
            rep_marker.scale.x              = 0.05
            rep_marker.scale.y              = 0.1
            rep_marker.scale.z              = 0.15
            rep_marker.color.r              = 0.8
            rep_marker.color.g              = 0.2
            rep_marker.color.b              = 0.8
            rep_marker.color.a              = 1.0
            rep_marker.pose.orientation.w   = 1.0
            rep_marker.points               = [Point(x=0.0, y=0.0, z=0.0), rep_point]
            rep_marker.lifetime             = Duration(seconds=self.marker_lifetime).to_msg()
            rep_markerArray.markers.append(rep_marker)

        self.repForce_pub.publish(rep_markerArray)

    def compute_resultant(self, points):
        resultant = Point(x=0.0, y=0.0, z=0.0)
        for point in points:
            resultant.x += point.x
            resultant.y += point.y
            resultant.z += point.z
        return resultant

    def publish_repulsiveResultant(self):
        if self.rep_from == ALL_OBSTACLES:
            self.rep_resultant = self.compute_resultant(self.rep_points)
        else:  
            if self.rep_points:
                closest_rep_point     = self.rep_points[0]
                closest_rep_magnitude = math.sqrt(closest_rep_point.x**2 + closest_rep_point.y**2)
                for rep_point in self.rep_points:
                    rep_magnitude = math.sqrt(rep_point.x**2 + rep_point.y**2)
                    if rep_magnitude > closest_rep_magnitude:
                        closest_rep_point     = rep_point
                        closest_rep_magnitude = rep_magnitude
                self.rep_resultant = closest_rep_point
            else:
                self.rep_resultant = Point(x=0.0, y=0.0, z=0.0)

        self.rep_resultant.x /= NORM_FACTOR_1
        self.rep_resultant.y /= NORM_FACTOR_1
        self.rep_resultant.z /= NORM_FACTOR_1

        resultant_marker                    = Marker()
        resultant_marker.header.frame_id    = 'base_link'
        resultant_marker.header.stamp       = self.get_clock().now().to_msg()
        resultant_marker.type               = Marker.ARROW
        resultant_marker.ns                 = 'repulsive_resultant'
        resultant_marker.scale.x            = 0.05
        resultant_marker.scale.y            = 0.1
        resultant_marker.scale.z            = 0.15
        resultant_marker.color.r            = 1.0
        resultant_marker.color.g            = 0.0
        resultant_marker.color.b            = 0.0
        resultant_marker.color.a            = 1.0
        resultant_marker.pose.orientation.w = 1.0
        resultant_marker.points             = [Point(x=0.0, y=0.0, z=0.0), self.rep_resultant]
        self.resForce_pub.publish(resultant_marker)

    def publish_referenceSignal(self):
        reference_marker                    = Marker()
        reference_marker.header.frame_id    = 'base_link'
        reference_marker.header.stamp       = self.get_clock().now().to_msg()
        reference_marker.type               = Marker.ARROW
        reference_marker.scale.x            = 0.05
        reference_marker.scale.y            = 0.1
        reference_marker.scale.z            = 0.15
        reference_marker.color.r            = 0.0
        reference_marker.color.g            = 0.0
        reference_marker.color.b            = 1.0
        reference_marker.color.a            = 1.0
        reference_marker.pose.orientation.w = 1.0
        reference_marker.points             = [Point(x=0.0, y=0.0, z=0.0), self.ref_signal]
        self.refSignal_pub.publish(reference_marker)

    def publish_finalVelocity_marker(self):
        """
        Computes the autonomous potential field vector blending and updates 
        RViz visualizations.
        """
        # Additive blend calculation: human intent + obstacle repulsion fields
        self.vfinal_signal = self.compute_resultant([self.ref_signal, self.rep_resultant])

        self.vfinal_signal.x /= NORM_FACTOR_2
        self.vfinal_signal.y /= NORM_FACTOR_2
        self.vfinal_signal.z /= NORM_FACTOR_2

        # Clamp calculations within standard unit bounds [-1.0, 1.0]
        self.vfinal_signal.x = max(-1.0, min(1.0, self.vfinal_signal.x))
        self.vfinal_signal.y = max(-1.0, min(1.0, self.vfinal_signal.y))

        # Publish the marker for Rviz visualization
        vfinal_marker                    = Marker()
        vfinal_marker.header.frame_id    = 'base_link'
        vfinal_marker.header.stamp       = self.get_clock().now().to_msg()
        vfinal_marker.type               = Marker.ARROW
        vfinal_marker.scale.x            = 0.05
        vfinal_marker.scale.y            = 0.1
        vfinal_marker.scale.z            = 0.15
        vfinal_marker.color.r            = 0.0
        vfinal_marker.color.g            = 1.0
        vfinal_marker.color.b            = 0.0
        vfinal_marker.color.a            = 1.0
        vfinal_marker.pose.orientation.w = 1.0
        vfinal_marker.points             = [Point(x=0.0, y=0.0, z=0.0), self.vfinal_signal]
        self.vfinal_marker_pub.publish(vfinal_marker)

        # Debug reporting
        ref_mag    = math.sqrt(self.ref_signal.x**2    + self.ref_signal.y**2)
        rep_mag    = math.sqrt(self.rep_resultant.x**2 + self.rep_resultant.y**2)
        vfinal_mag = math.sqrt(self.vfinal_signal.x**2 + self.vfinal_signal.y**2)

        if self.rho_values:
            rho_min = min(self.rho_values)
            rho_lines = f"\n  rho (nearest) = {rho_min:.3f} m"
        else:
            rho_lines = "\n  rho (nearest) = n/a"

        self.get_logger().info(
            f"\n"
            f"  DM Switch: {'ACTIVE' if self.deadman_switch == 1 else 'OFF'}\n"
            f"  ref    : ({self.ref_signal.x:+.3f}, {self.ref_signal.y:+.3f}) | mag_ref={ref_mag:.3f}\n"
            f"  rep    : ({self.rep_resultant.x:+.3f}, {self.rep_resultant.y:+.3f}) | mag_rep={rep_mag:.3f}\n"
            f"  vfinal : ({self.vfinal_signal.x:+.3f}, {self.vfinal_signal.y:+.3f}) | mag_vFi={vfinal_mag:.3f}"
            f"{rho_lines}"
        )

    def publish_vfinal_joy(self):
        """
        Populates and publishes the unified Joy message structure.
        Ensures hardware safety cutoff by checking the deadman switch state.
        """
        vfinal_joy_axes = list(self.vfinal_joy.axes)
        if len(vfinal_joy_axes) < 2:
            vfinal_joy_axes = [0.0, 0.0] + vfinal_joy_axes[2:]
        
        # Apply physical deadman safety gating to output fields
        if self.deadman_switch == 1:
            vfinal_joy_axes[0] = self.vfinal_signal.y   
            vfinal_joy_axes[1] = self.vfinal_signal.x   
        else:
            # Safe hardware immediate drop to neutral if L2 switch is released
            vfinal_joy_axes[0] = 0.0
            vfinal_joy_axes[1] = 0.0
            
        self.vfinal_joy.axes = tuple(vfinal_joy_axes)
        self.vfinal_joy_pub.publish(self.vfinal_joy)

    def publish_obstacles(self):
        self.roi1_ranges = [5.5] * len(self.base_r_values)

        points       = []   
        markerNumber = 0    

        for j in range(len(self.base_r_values)):
            if self.base_r_values[j] < self.marker_threshold_range:
                r     = self.base_r_values[j]
                theta = self.base_theta_values[j]
                marker_point   = Point()
                marker_point.x = r * math.cos(theta)
                marker_point.y = r * math.sin(theta)
                marker_point.z = 0.0
                points.append(marker_point)
            else:
                if points:
                    marker                      = Marker()
                    marker.header.frame_id      = 'base_link'
                    marker.header.stamp         = self.get_clock().now().to_msg()
                    marker.ns                   = 'thresholded_laserScan'
                    marker.id                   = markerNumber
                    marker.type                 = Marker.LINE_STRIP
                    marker.action               = Marker.ADD
                    marker.scale.x              = 0.02
                    marker.color.r              = 1.0
                    marker.color.g              = 0.0
                    marker.color.b              = 0.0
                    marker.color.a              = 1.0
                    marker.pose.orientation.w   = 1.0
                    marker.points               = points
                    marker.lifetime             = Duration(seconds=self.marker_lifetime).to_msg()
                    self.marker_pub.publish(marker)

                    self.record_closestPoint(points)
                    centroid = self.compute_centroid(points)
                    self.compute_roi1(points, centroid)
                    points        = []
                    markerNumber += 1

        if points:
            marker                      = Marker()
            marker.header.frame_id      = 'base_link'
            marker.header.stamp         = self.get_clock().now().to_msg()
            marker.ns                   = 'thresholded_laserScan'
            marker.id                   = markerNumber
            marker.type                 = Marker.LINE_STRIP
            marker.action               = Marker.ADD
            marker.scale.x              = 0.02
            marker.color.r              = 1.0
            marker.color.g              = 0.0
            marker.color.b              = 0.0
            marker.color.a              = 1.0
            marker.pose.orientation.w   = 1.0
            marker.points               = points
            marker.lifetime             = Duration(seconds=self.marker_lifetime).to_msg()
            self.marker_pub.publish(marker)

            self.record_closestPoint(points)
            centroid = self.compute_centroid(points)
            self.compute_roi1(points, centroid)

    # =============================================================================
    # Main loop (250 Hz timer callback)
    # =============================================================================
    def main_loop(self):
        if not self.displayLaserSpecs_once and self.front_ranges:
            self.displayLaserSpecs()
            self.displayLaserSpecs_once = True

        if not self.front_ranges:
            return

        self.centroids    = []
        self.closestPoints = []

        self.publish_transformed_pointCloud()
        self.publish_obstacles()       

        self.publish_centroids()
        self.publish_roi1()

        self.publish_potentialFields()        
        self.publish_repulsiveResultant()     

        self.publish_referenceSignal()        

        self.publish_finalVelocity_marker()   # Computes Twist and maps commands directly
        self.publish_vfinal_joy()             


def main(args=None):
    rclpy.init(args=sys.argv)
    node = SharedAutonomyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()