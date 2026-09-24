#!/usr/bin/env python3
# =================================================================================================
# haptic_audio_core.py  —  Pure-Python, ROS-FREE audio-haptic engine.
#
# WHY THIS FILE EXISTS
#   The audio rendering (48 kHz sine + per-sample amplitude smoothing), the force-shaping curves,
#   and the L/R directional panning have NOTHING to do with ROS. Kept here ONCE so that both the
#   ROS1 (rospy) and ROS2 (rclpy) shells import the identical engine — no duplicated audio logic,
#   no drift between the two robots. Same pattern as apcbf_core.py for the CBF math.
#
# HOW THE SHELLS USE IT
#   core = HapticAudioEngine(device="pulse", ...)     # construct with machine-specific device
#   core.start()                                       # opens + starts the audio stream
#   core.set_targets(distance_dm, direction_deg)       # called from each ROS subscriber callback
#   core.stop()                                        # glide to zero, close (no click)
#
#   The ONLY thing the ROS side does is unpack its own message and call set_targets(). Everything
#   below is ROS-agnostic and identical across ROS1/ROS2.
#
# UNITS
#   distance is in DECIMETERS (dm) to match the original node's tuning (min_dst=6 -> 0.6 m,
#   max_dst=10 -> 1.0 m). The shells convert their native meters -> dm before calling set_targets.
#   direction is in DEGREES (obstacle bearing): panning uses sin(direction).
# =================================================================================================

import math
import numpy as np

try:
    import sounddevice as sd
    _SD_IMPORT_ERROR = None
except Exception as e:                # allow import for unit tests on machines without audio
    sd = None
    _SD_IMPORT_ERROR = e


class HapticAudioEngine:
    def __init__(self,
                 device="pulse",
                 channels=4,
                 haptic_channels=(2, 3),   # DualSense USB-audio haptic channels (L, R)
                 speaker_channels=(0, 1),  # DualSense USB-audio speaker/headphone channels.
                                           # NOTE: the internal mono speaker is fed from the RIGHT
                                           # channel (index 1); we write both so either routing works.
                 fs=48000,
                 freq=60.0,                # actuator drive frequency (Hz)
                 beep_freq=880.0,          # rear-warning tone pitch (Hz)
                 master=0.8,               # overall amplitude cap
                 min_dst=6.0,              # dm: force SATURATES at/below this (0.6 m)
                 max_dst=10.0,             # dm: force BEGINS at/below this (1.0 m)
                 force_func=2,             # 1=linear, 2=quadratic, 3=exp, 4=log
                 alpha=1.0,
                 smooth=0.0008,            # per-sample glide coefficient (~25 ms)
                 blocksize=256):
        # ---- tuning ----
        self.min_dst = float(min_dst)
        self.max_dst = float(max_dst)
        self.force_func = int(force_func)
        self.alpha = float(alpha)
        self.freq = float(freq)
        self.master = float(master)

        # ---- audio config (machine/device specific -> passed in, NOT hardcoded) ----
        self.device = device
        self.channels = int(channels)
        self.chL, self.chR = haptic_channels
        self.spkA, self.spkB = speaker_channels
        self.fs = int(fs)
        self.blocksize = int(blocksize)
        self.smooth = float(smooth)

        # ---- runtime state ----
        self.phase = 0
        self.ampL_current = 0.0
        self.ampR_current = 0.0
        self.ampL_target = 0.0
        self.ampR_target = 0.0
        self.stream = None

        # ---- rear-warning beep state (speaker channels) ----
        self.beep_freq = float(beep_freq)
        self.beep_amp_current = 0.0
        self.beep_amp_target = 0.0        # 0 = silent; set via set_beep()
        self.pulse_rate = 4.0             # beeps per second (rate encodes urgency)
        self.pulse_phase = 0.0            # accumulated so the rhythm survives block boundaries

    # ============================ force shaping (verbatim logic) ============================
    def norm_x(self, x):
        return (self.max_dst - x) / (self.max_dst - self.min_dst)

    def haptic_force(self, x):
        if self.force_func == 1:
            return self.norm_x(x) #just norm
        elif self.force_func == 2:
            self.alpha = 10
            return self.alpha * self.norm_x(x) ** 2 #quadratic
        elif self.force_func == 3:
            z = 1 - self.norm_x(x)
            return 1 / math.exp(self.alpha * (z ** 2) * math.pi)
        elif self.force_func == 4:
            z = max(1e-6, 1 - self.norm_x(x))     # avoid log(0)
            return -math.log10(z)
        return 0.0

    def get_force(self, x):
        if x <= self.min_dst:
            return 1.0                 # saturate when very close (the corrected behavior)
        if x >= self.max_dst:
            return 0.0
        return min(1.0, max(0.0, self.haptic_force(x)))

    def directional(self, angle_deg):
        a = math.radians(angle_deg)
        right = 0.5 + math.sin(a) / 2
        left = 0.5 - math.sin(a) / 2
        return left, right

    # ============================ the ONLY interface the ROS side calls ============================
    def set_targets(self, distance_dm, direction_deg):
        """Given nearest-obstacle distance (dm) and bearing (deg), set L/R amplitude targets.
        Thread-safe enough for this use: the audio callback only READS the targets."""
        force = self.get_force(distance_dm)
        left, right = self.directional(direction_deg)
        self.ampL_target = self.master * force * left
        self.ampR_target = self.master * force * right

    def set_beep(self, amp, rate=None):
        """Drive the rear-warning beep on the speaker channels.
        amp:  0..1 target amplitude (0 = off; glides, so no click).
        rate: beeps per second (optional; keeps last rate if None)."""
        self.beep_amp_target = max(0.0, min(1.0, float(amp)))
        if rate is not None:
            self.pulse_rate = max(0.5, float(rate))

    def silence(self):
        """Command zero output (targets glide to 0 in the callback)."""
        self.ampL_target = 0.0
        self.ampR_target = 0.0
        self.beep_amp_target = 0.0

    # ============================ audio rendering (verbatim logic) ============================
    def _audio_callback(self, outdata, frames, time_info, status):
        t = (self.phase + np.arange(frames)) / self.fs
        sine = np.sin(2 * np.pi * self.freq * t)

        # rear-warning beep: tone gated by a square pulse envelope (50% duty cycle);
        # pulse phase accumulates across blocks so the rhythm doesn't stutter.
        tone = np.sin(2 * np.pi * self.beep_freq * t)
        pulse_t = self.pulse_phase + np.arange(frames) * (self.pulse_rate / self.fs)
        gate = ((pulse_t % 1.0) < 0.5).astype(np.float32)
        self.pulse_phase = pulse_t[-1] % 1.0
        self.phase += frames

        ampsL = np.empty(frames)
        ampsR = np.empty(frames)
        ampsB = np.empty(frames)
        aL, aR = self.ampL_current, self.ampR_current
        aB = self.beep_amp_current
        # for i in range(frames):
        #     aL += self.smooth * (self.ampL_target - aL)
        #     aR += self.smooth * (self.ampR_target - aR)
        #     aB += self.smooth * (self.beep_amp_target - aB)
        #     ampsL[i] = aL
        #     ampsR[i] = aR
        #     ampsB[i] = aB
        # self.ampL_current, self.ampR_current = aL, aR
        # self.beep_amp_current = aB

        decay = (1.0 - self.smooth) ** np.arange(1, frames + 1)
        ampsL = self.ampL_target + (self.ampL_current - self.ampL_target) * decay
        ampsR = self.ampR_target + (self.ampR_current - self.ampR_target) * decay
        ampsB = self.beep_amp_target + (self.beep_amp_current - self.beep_amp_target) * decay
        self.ampL_current = float(ampsL[-1])
        self.ampR_current = float(ampsR[-1])
        self.beep_amp_current = float(ampsB[-1])

        outdata[:] = 0.0
        beep = (ampsB * gate * tone).astype(np.float32)
        outdata[:, self.spkA] = beep
        outdata[:, self.spkB] = beep     # speaker is fed from the right channel (index 1)
        outdata[:, self.chL] = (ampsL * sine).astype(np.float32)
        outdata[:, self.chR] = (ampsR * sine).astype(np.float32)

    # ============================ lifecycle ============================
    # def start(self):
    #     if sd is None:
    #         raise RuntimeError("sounddevice not available in this environment")
    #     self.stream = sd.OutputStream(
    #         samplerate=self.fs, channels=self.channels, blocksize=self.blocksize,
    #         device=self.device, callback=self._audio_callback)
    #     self.stream.start()

    def start(self):
        if sd is None:
            raise RuntimeError(f"sounddevice not available: {_SD_IMPORT_ERROR}")
        self.stream = sd.OutputStream(
            samplerate=self.fs, channels=self.channels, blocksize=self.blocksize,
            device=self.device, callback=self._audio_callback)
        self.stream.start()

    def stop(self, glide_seconds=0.2):
        self.silence()
        if self.stream is not None:
            try:
                import time
                time.sleep(glide_seconds)     # let amplitude glide to zero -> no click
            except Exception:
                pass
            self.stream.stop()
            self.stream.close()
            self.stream = None


# ============================ standalone self-test (no ROS, no device) ============================
if __name__ == "__main__":
    # Verify force shaping and panning without opening an audio device.
    eng = HapticAudioEngine()
    print("force curve (dm -> force):")
    for d in (11, 10, 9, 8, 7, 6, 5):
        print(f"  {d:2d} dm -> {eng.get_force(d):.3f}")
    print("panning (deg -> L,R):")
    for ang in (-90, -45, 0, 45, 90):
        l, r = eng.directional(ang)
        print(f"  {ang:+3d} deg -> L={l:.2f} R={r:.2f}")
    print("set_targets(7 dm, +60 deg):")
    eng.set_targets(7, 60)
    print(f"  ampL_target={eng.ampL_target:.3f}  ampR_target={eng.ampR_target:.3f}")
