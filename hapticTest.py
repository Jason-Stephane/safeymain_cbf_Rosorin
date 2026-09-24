# import os
# os.environ["PULSE_SINK"] = "alsa_output.usb-Sony_Interactive_Entertainment_Wireless_Controller-00.analog-surround-40"

# fs = 48000
# dev = "pulse"  # or find the DualSense index via sd.query_devices()
# f, amp = 60.0, 0.8          # frequency (Hz), amplitude 0..1
# t = np.arange(fs) / fs
# sine = amp * np.sin(2 * np.pi * f * t)

# buf = np.zeros((fs, 4), dtype=np.float32)
# buf[:, 2] = sine   # left actuator
# buf[:, 3] = sine   # right actuator

# import os; os.environ["PULSE_SINK"] = "alsa_output.usb-Sony_Interactive_Entertainment_Wireless_Controller-00.analog-surround-40"
# sd.play(buf, fs, device=dev, blocking=True)

# # Left grip only
# buf = np.zeros((fs, 4), dtype=np.float32)
# buf[:, 2] = sine
# sd.play(buf, fs, blocking=True)

# # Right grip only
# buf = np.zeros((fs, 4), dtype=np.float32)
# buf[:, 3] = sine
# sd.play(buf, fs, blocking=True)


#############################################################################################################
# rumble simulating distance
# import os
# os.environ["PULSE_SINK"] = "alsa_output.usb-Sony_Interactive_Entertainment_Wireless_Controller-00.analog-surround-40"
# import numpy as np, sounddevice as sd, time

# fs = 48000
# freq = 60.0
# phase = 0.0
# amp_current = 0.0          # what's playing right now
# amp_target = 0.0           # updated from outside (e.g., ROS callback)
# smooth = 0.0008            # per-sample smoothing factor (~25 ms time constant)

# def callback(outdata, frames, time_info, status):
#     global phase, amp_current
#     t = (phase + np.arange(frames)) / fs
#     sine = np.sin(2 * np.pi * freq * t)
#     phase += frames                      # carry phase into the next block

#     # exponential glide of amplitude toward the target, sample by sample
#     amps = np.empty(frames)
#     a = amp_current
#     for i in range(frames):
#         a += smooth * (amp_target - a)
#         amps[i] = a
#     amp_current = a

#     outdata[:] = 0.0
#     outdata[:, 2] = (amps * sine).astype(np.float32)
#     outdata[:, 3] = (amps * sine).astype(np.float32)

# stream = sd.OutputStream(samplerate=fs, channels=4, callback=callback,
#                          blocksize=256, device="pulse")
# stream.start()

# # --- simulate an approaching then retreating obstacle ---
# for d in list(np.linspace(2.0, 0.2, 50)) + list(np.linspace(0.2, 2.0, 50)):
#     amp_target = float(np.clip((2.0 - d) / 1.8, 0, 1)**2)
#     time.sleep(0.1)                      # pretend this is your 10 Hz LIDAR rate

# stream.stop(); stream.close()



#left and right distance rumble

"""import os
os.environ["PULSE_SINK"] = "alsa_output.usb-Sony_Interactive_Entertainment_Wireless_Controller-00.analog-surround-40"
import numpy as np, sounddevice as sd, time

fs = 48000
freq = 60.0
phase = 0.0
ampL_current = 0.0
ampR_current = 0.0
ampL_target = 0.0          # set these two from outside (ROS callback / test loop)
ampR_target = 0.0
smooth = 0.0008            # ~25 ms glide

def callback(outdata, frames, time_info, status):
    global phase, ampL_current, ampR_current
    t = (phase + np.arange(frames)) / fs
    sine = np.sin(2 * np.pi * freq * t)
    phase += frames

    ampsL = np.empty(frames); ampsR = np.empty(frames)
    aL, aR = ampL_current, ampR_current
    for i in range(frames):
        aL += smooth * (ampL_target - aL)
        aR += smooth * (ampR_target - aR)
        ampsL[i] = aL; ampsR[i] = aR
    ampL_current, ampR_current = aL, aR

    outdata[:] = 0.0
    outdata[:, 2] = (ampsL * sine).astype(np.float32)   # left grip
    outdata[:, 3] = (ampsR * sine).astype(np.float32)   # right grip

stream = sd.OutputStream(samplerate=fs, channels=4, callback=callback,
                         blocksize=256, device="pulse")
stream.start()

# ---- Demo 1: left only, then right only ----
print("left grip...")
ampL_target, ampR_target = 0.8, 0.0
time.sleep(2)
print("right grip...")
ampL_target, ampR_target = 0.0, 0.8
time.sleep(2)

# ---- Demo 2: obstacle sweeping around the robot, left → front → right ----
print("panning sweep...")
d = 0.5                                   # fixed close distance
closeness = np.clip((2.0 - d) / 1.8, 0, 1)**2
for theta in np.linspace(np.pi/2, -np.pi/2, 100):   # +90° (left) to -90° (right)
    pan = (1 + np.sin(theta)) / 2          # 1 = fully left, 0 = fully right
    ampL_target = closeness * pan
    ampR_target = closeness * (1 - pan)
    time.sleep(0.05)

ampL_target = ampR_target = 0.0
time.sleep(0.5)
stream.stop(); stream.close()

"""


import os
os.environ["PULSE_SINK"] = "alsa_output.usb-Sony_Interactive_Entertainment_Wireless_Controller-00.analog-surround-40"
import numpy as np, sounddevice as sd, time

fs = 48000
freq = 60.0
phase = 0.0
ampL_current = ampR_current = 0.0
ampL_target = ampR_target = 0.0
smooth = 0.0008

def callback(outdata, frames, time_info, status):
    global phase, ampL_current, ampR_current
    t = (phase + np.arange(frames)) / fs
    sine = np.sin(2 * np.pi * freq * t)
    phase += frames

    ampsL = np.empty(frames); ampsR = np.empty(frames)
    aL, aR = ampL_current, ampR_current
    for i in range(frames):
        aL += smooth * (ampL_target - aL)
        aR += smooth * (ampR_target - aR)
        ampsL[i] = aL; ampsR[i] = aR
    ampL_current, ampR_current = aL, aR

    outdata[:] = 0.0
    outdata[:, 2] = (ampsL * sine).astype(np.float32)   # left grip
    outdata[:, 3] = (ampsR * sine).astype(np.float32)   # right grip

def distance_to_closeness(d, d_max=2.0, d_min=0.2):
    return float(np.clip((d_max - d) / (d_max - d_min), 0.0, 1.0)**2)

def set_haptics(d, theta):
    """d: distance to obstacle (m). theta: bearing, +pi/2 = left, -pi/2 = right, 0 = ahead."""
    global ampL_target, ampR_target
    closeness = distance_to_closeness(d)
    pan = (1 + np.sin(theta)) / 2          # 1 = fully left, 0 = fully right
    ampL_target = closeness * pan
    ampR_target = closeness * (1 - pan)

stream = sd.OutputStream(samplerate=fs, channels=4, callback=callback,
                         blocksize=256, device="pulse")
stream.start()
dt = 0.05   # 20 Hz update, like a scan rate

# ---- Scenario 1: obstacle approaches from the LEFT (2.0 m -> 0.2 m) ----
print("1) approaching from the left...")
for d in np.linspace(2.0, 0.2, 80):
    set_haptics(d, theta=np.pi/2)
    time.sleep(dt)
set_haptics(5.0, 0); time.sleep(1.0)        # obstacle gone, fade out

# ---- Scenario 2: obstacle approaches from the RIGHT ----
print("2) approaching from the right...")
for d in np.linspace(2.0, 0.2, 80):
    set_haptics(d, theta=-np.pi/2)
    time.sleep(dt)
set_haptics(5.0, 0); time.sleep(1.0)

# ---- Scenario 3: obstacle pans L -> R -> L while steadily approaching ----
print("3) panning left-right-left while closing in...")
n = 240                                      # 12 seconds
distances = np.linspace(2.0, 0.2, n)         # steady approach
bearings  = (np.pi/2) * np.sin(2*np.pi * np.arange(n)/n * 1.5)  # 1.5 full L-R-L cycles
for d, th in zip(distances, bearings):
    set_haptics(d, th)
    time.sleep(dt)

set_haptics(5.0, 0); time.sleep(1.0)
stream.stop(); stream.close()