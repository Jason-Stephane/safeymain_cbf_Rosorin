#!/usr/bin/env python3
#ROS2 node to record CBF data (h values) and velocity commands (Twist) into a CSV file.
# The CSV file will include timestamps, h values, and velocity commands for analysis.
# Run this node alongside your existing ROS2 setup to capture the necessary data.
#Run : python3 cbf_recorder.py


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import csv
import os
from datetime import datetime

class CBFDataRecorder(Node):
    def __init__(self, user_id, save_dir):
        super().__init__('cbf_data_recorder')

        # Cache for the latest h value
        self.latest_h = 1.0  

        # Ensure the target directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Capture computer local start time
        self.start_time = datetime.now()
        time_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(save_dir, f"cbf_data_ID-{user_id}_{time_str}.csv")

        # Open the CSV file and write the metadata and headers
        self.file = open(self.filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)
        
        # Write the start time at the top of the file
        self.csv_writer.writerow(['# Session Start Time (Local)', self.start_time.strftime('%Y-%m-%d %H:%M:%S.%f')])
        
        # Write standard headers including ros_time_sec
        self.csv_writer.writerow([
            'ros_time_raw_nanosec',
            'ros_time_sec',
            'computer_local_time', 
            'h', 
            'vx', 
            'vy', 
            'wz'
        ])

        # Subscribers mapping to your cbf_joy10.py outputs
        self.create_subscription(Float32, '/cbf/h', self.h_callback, 10)
        self.create_subscription(Twist, '/controller/cmd_vel', self.cmd_callback, 10)

        self.get_logger().info(f"Recording started for ID: {user_id}")
        self.get_logger().info(f"Saving data to: {self.filename}")
        self.get_logger().info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def h_callback(self, msg: Float32):
        """Cache the latest barrier value."""
        self.latest_h = msg.data

    def cmd_callback(self, msg: Twist):
        """Trigger a CSV write every time a new velocity command is published."""
        # Record time
        ros_time_raw_nanosec = self.get_clock().now().nanoseconds
        ros_time_sec = ros_time_raw_nanosec / 1e9  # Convert nanoseconds to standard ROS seconds
        computer_local_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        # Extract velocities
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        # Write the row
        self.csv_writer.writerow([
            ros_time_raw_nanosec,
            ros_time_sec,
            computer_local_time, 
            self.latest_h, 
            vx, 
            vy, 
            wz
        ])

    def destroy_node(self):
        """Ensure the CSV file is safely closed and log the end time when the node shuts down."""
        # Capture computer local end time
        end_time = datetime.now()
        
        # Append the end time at the bottom of the CSV
        self.csv_writer.writerow(['# Session End Time (Local)', end_time.strftime('%Y-%m-%d %H:%M:%S.%f')])
        
        self.file.close()
        self.get_logger().info(f"Recording stopped at {end_time.strftime('%Y-%m-%d %H:%M:%S')}.")
        self.get_logger().info(f"File saved at {self.filename}")
        super().destroy_node()

def main(args=None):
    # Ask for the ID before initializing ROS
    print("=== CBF Data Recorder ===")
    user_id = input("Please enter your ID: ")
    
    # Target directory
    save_dir = "/home/jasonstephane/safety_main_cbf"

    rclpy.init(args=args)
    node = CBFDataRecorder(user_id, save_dir)

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
