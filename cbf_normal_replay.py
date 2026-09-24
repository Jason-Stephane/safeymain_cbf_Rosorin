#!/usr/bin/env python3

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_latest_csv(user_id, directory):
    """Finds the most recently created CSV file for the given user ID."""
    search_pattern = os.path.join(directory, f"cbf_data_ID-{user_id}_*.csv")
    files = glob.glob(search_pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def main():
    print("=== CBF Data Real-Time Replay ===")
    user_id = input("Enter the User ID to replay: ")
    
    target_dir = "/home/jasonstephane/safety_main_cbf"
    
    latest_file = get_latest_csv(user_id, target_dir)
    if latest_file is None:
        print(f"Error: No CSV files found for ID '{user_id}' in {target_dir}")
        return
        
    print(f"Loading data from: {latest_file}")
    
    # Read the CSV, ignoring the metadata rows starting with '#'
    df = pd.read_csv(latest_file, comment='#')
    
    if len(df) < 2:
        print("Not enough data in this file to replay.")
        return

    # Normalize time so the graph starts at 0 seconds
    t_start = df['ros_time_sec'].iloc[0]
    df['t_norm'] = df['ros_time_sec'] - t_start

    # Set up the Matplotlib 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # Real-time playback settings
    interval_ms = 50  # Update the graph every 50 milliseconds (20 FPS)
    max_time = df['t_norm'].iloc[-1]
    total_frames = int((max_time * 1000) / interval_ms) + 10 # Add a small buffer at the end

    def animate(i):
        # Calculate the exact elapsed time in seconds for this frame
        current_sim_time = i * (interval_ms / 1000.0)
        
        # Filter the dataframe to only include rows that happened up to this exact time
        current_df = df[df['t_norm'] <= current_sim_time]
        
        if current_df.empty:
            return

        # Convert Pandas Series to NumPy arrays to fix the Matplotlib indexing error
        t = current_df['t_norm'].to_numpy()
        h = current_df['h'].to_numpy()
        d = current_df['delta'].to_numpy()
        vx = current_df['vx'].to_numpy()
        vy = current_df['vy'].to_numpy()
        wz = current_df['wz_angular'].to_numpy()

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

    print(f"Starting real-time replay (Total duration: {max_time:.1f} seconds)...")
    print("Close the window to exit.")
    
    # Run the animation
    ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=interval_ms, repeat=False)
    
    plt.show()

if __name__ == '__main__':
    main()