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
    print("=== CBF Data Replay ===")
    user_id = input("Enter the User ID to replay: ")
    
    target_dir = "/home/jasonstephane/safety_main_cbf"
    
    latest_file = get_latest_csv(user_id, target_dir)
    if latest_file is None:
        print(f"Error: No CSV files found for ID '{user_id}' in {target_dir}")
        return
        
    print(f"Loading data from: {latest_file}")
    
    # Read the CSV, ignoring the metadata rows starting with '#'
    df = pd.read_csv(latest_file, comment='#')
    
    if len(df) < 10:
        print("Not enough data in this file to replay.")
        return

    # Normalize time so the graph starts at 0 seconds
    t_start = df['ros_time_sec'].iloc[0]
    df['t_norm'] = df['ros_time_sec'] - t_start

    # Set up the Matplotlib 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # Calculate how many rows to reveal per frame so the replay takes ~10 seconds at 20 FPS
    frames_per_update = max(1, len(df) // 200)
    total_frames = (len(df) // frames_per_update) + 1

    def animate(i):
        # Calculate the chunk of data to display up to this frame
        end_idx = min((i + 1) * frames_per_update, len(df))
        current_df = df.iloc[:end_idx]
        
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
        axs[0, 0].set_xlim(0, df['t_norm'].iloc[-1]) # Keep x-axis fixed to total duration
        axs[0, 0].legend(loc='upper right')
        axs[0, 0].grid(True, linestyle='dotted', alpha=0.7)

        # 2. vx, vy vs time (Top Right)
        axs[0, 1].clear()
        axs[0, 1].set_title(f"Participant {user_id}: Linear Velocities (vx, vy)", fontweight='bold', fontsize=12)
        axs[0, 1].plot(t, vx, color='tab:green', label='Forward/Back (vx)')
        axs[0, 1].plot(t, vy, color='tab:red', label='Strafe (vy)')
        axs[0, 1].set_ylabel("Speed (m/s)")
        axs[0, 1].set_xlabel("Elapsed Time (seconds)")
        axs[0, 1].set_xlim(0, df['t_norm'].iloc[-1])
        axs[0, 1].legend(loc='upper right')
        axs[0, 1].grid(True, linestyle='dotted', alpha=0.7)

        # 3. delta vs time (Bottom Left)
        axs[1, 0].clear()
        axs[1, 0].set_title(f"Participant {user_id}: Velocity Disagreement (Delta)", fontweight='bold', fontsize=12)
        axs[1, 0].plot(t, d, color='tab:orange', label='Delta Magnitude')
        axs[1, 0].set_ylabel("Delta Magnitude")
        axs[1, 0].set_xlabel("Elapsed Time (seconds)")
        axs[1, 0].set_xlim(0, df['t_norm'].iloc[-1])
        axs[1, 0].grid(True, linestyle='dotted', alpha=0.7)

        # 4. wz_angular vs time (Bottom Right)
        axs[1, 1].clear()
        axs[1, 1].set_title(f"Participant {user_id}: Angular Velocity (wz)", fontweight='bold', fontsize=12)
        axs[1, 1].plot(t, wz, color='tab:purple', label='Rotation (wz_angular)')
        axs[1, 1].set_ylabel("Speed (rad/s)")
        axs[1, 1].set_xlabel("Elapsed Time (seconds)")
        axs[1, 1].set_xlim(0, df['t_norm'].iloc[-1])
        axs[1, 1].legend(loc='upper right')
        axs[1, 1].grid(True, linestyle='dotted', alpha=0.7)

    print("Starting replay... Close the window when finished.")
    
    # Run the animation
    ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=50, repeat=False)
    
    plt.show()

if __name__ == '__main__':
    main()