#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import csv
import os
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CBFDataRecorder(Node):
    def __init__(self, user_id, save_dir):
        super().__init__('cbf_data_recorder')

        self.latest_h = 1.0  
        self.latest_delta = 0.0

        # Memory storage for live plotting
        self.plot_lock = threading.Lock()
        self.t_data = []
        self.h_data = []
        self.delta_data = []
        self.vx_data = []
        self.vy_data = []
        self.wz_data = []
        self.first_ros_time = None

        # Ensure the target directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Capture computer local start time
        self.session_start = datetime.now()
        time_str = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Files are saved in the exact same directory
        self.filename = os.path.join(save_dir, f"cbf_data_ID-{user_id}_{time_str}.csv")
        self.img_filename = self.filename.replace('.csv', '.jpg')

        # Open the CSV file
        self.file = open(self.filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)
        
        self.csv_writer.writerow(['# Session Start Time (Local)', self.session_start.strftime('%Y-%m-%d %H:%M:%S.%f')])
        self.csv_writer.writerow([
            'ros_time_raw_nanosec',
            'ros_time_sec',
            'row_start_time', 
            'row_end_time',
            'h', 
            'vx', 
            'vy', 
            'wz_angular',
            'delta'
        ])

        # Subscribers
        self.create_subscription(Float32, '/cbf/h', self.h_callback, 10)
        self.create_subscription(Float32, '/haptic/cbf_delta', self.delta_callback, 10)
        self.create_subscription(Twist, '/controller/cmd_vel', self.cmd_callback, 10)

        self.get_logger().info(f"Recording started. Saving data to: {self.filename}")

    def h_callback(self, msg: Float32):
        self.latest_h = msg.data
        
    def delta_callback(self, msg: Float32):
        self.latest_delta = msg.data

    def cmd_callback(self, msg: Twist):
        row_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        ros_time_raw = self.get_clock().now().nanoseconds
        ros_time_sec = ros_time_raw / 1e9  

        vx = msg.linear.x
        vy = msg.linear.y
        wz_angular = msg.angular.z

        row_end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        # 1. Write to CSV
        self.csv_writer.writerow([
            ros_time_raw, ros_time_sec, row_start_time, row_end_time, 
            self.latest_h, vx, vy, wz_angular, self.latest_delta
        ])

        # 2. Save to memory for live plot (thread-safe)
        with self.plot_lock:
            if self.first_ros_time is None:
                self.first_ros_time = ros_time_sec
            
            self.t_data.append(ros_time_sec - self.first_ros_time)
            self.h_data.append(self.latest_h)
            self.delta_data.append(self.latest_delta)
            self.vx_data.append(vx)
            self.vy_data.append(vy)
            self.wz_data.append(wz_angular)

    def safe_shutdown(self):
        """Closes the file and logs the end time."""
        session_end = datetime.now()
        self.csv_writer.writerow(['# Session End Time (Local)', session_end.strftime('%Y-%m-%d %H:%M:%S.%f')])
        self.file.close()
        self.get_logger().info("Recording stopped. File closed securely.")


def ros_thread_function(node):
    """Runs the ROS 2 event loop in a background thread."""
    rclpy.spin(node)


def main(args=None):
    print("=== Live CBF Recorder & Grapher ===")
    user_id = input("Please enter your ID: ")
    save_dir = "/home/jasonstephane/safety_main_cbf"

    rclpy.init(args=args)
    node = CBFDataRecorder(user_id, save_dir)

    # Launch ROS 2 in a background thread
    ros_thread = threading.Thread(target=ros_thread_function, args=(node,), daemon=True)
    ros_thread.start()

    # --- Matplotlib Setup (Main Thread) ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    def animate(i):
        # Safely copy data from the ROS thread
        with node.plot_lock:
            if not node.t_data:
                return
            t = list(node.t_data)
            h = list(node.h_data)
            d = list(node.delta_data)
            vx = list(node.vx_data)
            vy = list(node.vy_data)
            wz = list(node.wz_data)

        # 1. h vs time (Top Left)
        axs[0, 0].clear()
        axs[0, 0].set_title(f"Participant {user_id}: Safety Boundary (h)", fontweight='bold', fontsize=12)
        axs[0, 0].plot(t, h, color='tab:blue', label='Safety Value (h)')
        axs[0, 0].axhline(0, color='red', linestyle='--', label='Safety Boundary (h=0)')
        axs[0, 0].set_ylabel("Safety Value (h)")
        axs[0, 0].set_xlabel("Elapsed Time (seconds)")
        axs[0, 0].legend(loc='upper right')
        axs[0, 0].grid(True, linestyle='dotted', alpha=0.7)

        # 2. vx, vy vs time (Top Right)
        axs[0, 1].clear()
        axs[0, 1].set_title(f"Participant {user_id}: Linear Velocities (vx, vy)", fontweight='bold', fontsize=12)
        axs[0, 1].plot(t, vx, color='tab:green', label='Forward/Back (vx)')
        axs[0, 1].plot(t, vy, color='tab:red', label='Strafe (vy)')
        axs[0, 1].set_ylabel("Speed (m/s)")
        axs[0, 1].set_xlabel("Elapsed Time (seconds)")
        axs[0, 1].legend(loc='upper right')
        axs[0, 1].grid(True, linestyle='dotted', alpha=0.7)

        # 3. delta vs time (Bottom Left)
        axs[1, 0].clear()
        axs[1, 0].set_title(f"Participant {user_id}: Velocity Disagreement (Delta)", fontweight='bold', fontsize=12)
        axs[1, 0].plot(t, d, color='tab:orange', label='Delta Magnitude')
        axs[1, 0].set_ylabel("Delta Magnitude")
        axs[1, 0].set_xlabel("Elapsed Time (seconds)")
        axs[1, 0].grid(True, linestyle='dotted', alpha=0.7)

        # 4. wz_angular vs time (Bottom Right)
        axs[1, 1].clear()
        axs[1, 1].set_title(f"Participant {user_id}: Angular Velocity (wz)", fontweight='bold', fontsize=12)
        axs[1, 1].plot(t, wz, color='tab:purple', label='Rotation (wz_angular)')
        axs[1, 1].set_ylabel("Speed (rad/s)")
        axs[1, 1].set_xlabel("Elapsed Time (seconds)")
        axs[1, 1].legend(loc='upper right')
        axs[1, 1].grid(True, linestyle='dotted', alpha=0.7)

    # Start the live animation
    ani = animation.FuncAnimation(fig, animate, interval=250, cache_frame_data=False)
    
    print("Graphing window is live. Close the window or press CTRL+C to save data and exit.")
    
    # ---------------------------------------------------------
    # Execution blocks here until the user closes the window or presses CTRL+C.
    # ---------------------------------------------------------
    try:
        plt.show() 
    except KeyboardInterrupt:
        print("\nCTRL+C detected. Shutting down gracefully...")
    finally:
        # ---------------------------------------------------------
        # This guaranteed shutdown sequence runs no matter how the script stops.
        # ---------------------------------------------------------
        print("\nSaving graph and shutting down ROS...")
        fig.savefig(node.img_filename, bbox_inches='tight', dpi=150)
        print(f"Graph saved to: {node.img_filename}")
        
        node.safe_shutdown()
        
        # Ensure ROS shuts down cleanly if it hasn't already
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()