#!/usr/bin/env python3
# haptic_core4.py  — ROS-FREE audio-haptic engine
# Changes vs haptic_core3.py:
#   1. set_multi_targets: fix corridor detection ordering bug (accumulate THEN check)
#   2. Corridor cue: replace ±15% modulation with a 1.5 Hz L/R alternation with 20% silence gap
#   3. set_disagreement(): new method for CBF legibility — intensity ∝ ||u_des - u_safe||
#      Uses the same master/force curve but bypasses distance entirely.

import math
import time
import numpy as np

try:
    import sounddevice as sd
    _SD_IMPORT_ERROR = None
except Exception as e:
    sd = None
    _SD_IMPORT_ERROR = e


class HapticAudioEngine:
    def __init__(self,
                 device="pulse",
                 channels=4,
                 haptic_channels=(2, 3),
                 speaker_channels=(0, 1),
                 fs=48000,
                 freq=60.0,
                 beep_freq=880.0,
                 master=0.2,
                 min_dst=6.0,
                 max_dst=10.0,
                 force_func=2,
                 alpha=1.0,
                 smooth=0.0008,
                 blocksize=256):
        self.min_dst   = float(min_dst)
        self.max_dst   = float(max_dst)
        self.force_func = int(force_func)
        self.alpha     = float(alpha)
        self.freq      = float(freq)
        self.master    = float(master)
        self.device    = device
        self.channels  = int(channels)
        self.chL, self.chR   = haptic_channels
        self.spkA, self.spkB = speaker_channels
        self.fs        = int(fs)
        self.blocksize = int(blocksize)
        self.smooth    = float(smooth)

        self.phase          = 0
        self.ampL_current   = 0.0
        self.ampR_current   = 0.0
        self.ampL_target    = 0.0
        self.ampR_target    = 0.0
        self.stream         = None

        self.beep_freq       = float(beep_freq)
        self.beep_amp_current = 0.0
        self.beep_amp_target  = 0.0
        self.pulse_rate      = 4.0
        self.pulse_phase     = 0.0

        # Corridor alternation state
        # Each "cycle" = 1/corridor_rate seconds split into left-on / gap / right-on / gap
        self._corridor_active  = False
        self._corridor_rate    = 1.5        # full L→R cycles per second
        self._corridor_gap     = 0.10       # fraction of half-cycle that's silent (0–0.5)

    # ========================= force shaping =========================
    def norm_x(self, x):
        return (self.max_dst - x) / (self.max_dst - self.min_dst)

    def haptic_force(self, x):
        if self.force_func == 1:
            return self.norm_x(x)
        elif self.force_func == 2:
            self.alpha = 10
            return self.alpha * self.norm_x(x) ** 2
        elif self.force_func == 3:
            z = 1 - self.norm_x(x)
            return 1 / math.exp(self.alpha * (z ** 2) * math.pi)
        elif self.force_func == 4:
            z = max(1e-6, 1 - self.norm_x(x))
            return -math.log10(z)
        return 0.0

    def get_force(self, x):
        if x <= self.min_dst:
            return 1.0
        if x >= self.max_dst:
            return 0.0
        return min(1.0, max(0.0, self.haptic_force(x)))

    def directional(self, angle_deg):
        a = math.radians(angle_deg)
        right = 0.5 + math.sin(a) / 2
        left  = 0.5 - math.sin(a) / 2
        return left, right

    # ========================= single obstacle =========================
    def set_targets(self, distance, direction_deg):
        self._corridor_active = False
        force = self.get_force(distance)
        left, right = self.directional(direction_deg)
        self.ampL_target = self.master * force * left
        self.ampR_target = self.master * force * right

    # ========================= CBF disagreement mode (NEW) =========================
    def set_disagreement(self, u_des, u_safe, direction_deg=0.0, max_delta=1.0):
        """
        Set haptic intensity proportional to how much the CBF filter corrected the command.

        Args:
            u_des:        desired velocity command — scalar (e.g. linear.x) or (vx, wz) tuple/list
            u_safe:       filtered/safe velocity command — same type as u_des
            direction_deg: bearing of the nearest obstacle, for panning (default 0 = ahead)
            max_delta:    the correction magnitude that maps to full intensity (tune this)
        """
        self._corridor_active = False

        # Support scalar or 2-vector commands
        if hasattr(u_des, '__len__'):
            delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(u_des, u_safe)))
        else:
            delta = abs(float(u_des) - float(u_safe))

        # Clamp and normalise: 0 → no vibration, max_delta → full intensity
        intensity = min(1.0, delta / max(1e-6, float(max_delta)))

        left, right = self.directional(direction_deg)
        self.ampL_target = self.master * intensity * left
        self.ampR_target = self.master * intensity * right

    # ========================= multi-obstacle =========================
    def set_multi_targets(self, obstacles):
        """
        Multiple obstacles → independent L/R haptic force.

        Each obstacle is assigned to its dominant channel based on bearing sign,
        with a narrow front-center zone (±front_half degrees) that feeds both
        channels equally.  This gives sharp L/R separation — an obstacle clearly
        on the left drives only the left actuator, and vice-versa.

        Corridor cue (walls on both sides):
          A 1.5 Hz left/right alternation with a 10 % silence gap — conspicuously
          different from the steady bilateral buzz of a single front wall.
        """
        if not obstacles:
            self._corridor_active = False
            self.ampL_target = 0.0
            self.ampR_target = 0.0
            return

        FRONT_HALF = 15.0   # degrees — obstacle within ±15° of centre feeds both channels

        # ---- 1. Hard L/R assignment with narrow centre crossover ----
        max_L = 0.0
        max_R = 0.0
        has_left_obstacle  = False
        has_right_obstacle = False

        for dist, bearing in obstacles:
            force = self.get_force(dist)
            if force <= 0.0:
                continue

            if bearing < -FRONT_HALF:
                # Left-side obstacle → left channel only
                max_L = max(max_L, force)
                has_left_obstacle = True
            elif bearing > FRONT_HALF:
                # Right-side obstacle → right channel only
                max_R = max(max_R, force)
                has_right_obstacle = True
            else:
                # Front-centre → both channels equally
                max_L = max(max_L, force)
                max_R = max(max_R, force)

        # ---- 2. Corridor detection ----
        if has_left_obstacle and has_right_obstacle and max_L > 0.05 and max_R > 0.05:
            self._corridor_active = True
            self._corridor_base_L = self.master * max_L
            self._corridor_base_R = self.master * max_R
            return

        # ---- 3. Normal (non-corridor) case ----
        self._corridor_active = False
        self.ampL_target = self.master * max_L
        self.ampR_target = self.master * max_R

    # ========================= beep =========================
    def set_beep(self, amp, rate=None):
        self.beep_amp_target = max(0.0, min(1.0, float(amp)))
        if rate is not None:
            self.pulse_rate = max(0.5, float(rate))

    def silence(self):
        self._corridor_active = False
        self.ampL_target = 0.0
        self.ampR_target = 0.0
        self.beep_amp_target = 0.0

    # ========================= audio callback =========================
    def _audio_callback(self, outdata, frames, time_info, status):
        t = (self.phase + np.arange(frames)) / self.fs
        sine = np.sin(2 * np.pi * self.freq * t)

        tone    = np.sin(2 * np.pi * self.beep_freq * t)
        pulse_t = self.pulse_phase + np.arange(frames) * (self.pulse_rate / self.fs)
        gate    = ((pulse_t % 1.0) < 0.5).astype(np.float32)
        self.pulse_phase = pulse_t[-1] % 1.0
        self.phase      += frames

        # ---- Corridor alternation (computed per-sample, no jitter) ----
        if self._corridor_active:
            # Each full cycle = 1/rate seconds split into 4 equal quarters:
            #   Q0: left ON  (1 - gap fraction)
            #   Q1: silence (gap)
            #   Q2: right ON (1 - gap fraction)
            #   Q3: silence (gap)
            now = time.monotonic()
            # per-sample time relative to now
            t_abs = now + np.arange(frames) / self.fs
            cycle_pos = (t_abs * self._corridor_rate * 2) % 2.0   # 0–2 ramps per L/R cycle
            half_active = 1.0 - self._corridor_gap                  # fraction of each half that's on

            gate_L = (cycle_pos < half_active).astype(np.float32)
            gate_R = ((cycle_pos >= 1.0) & (cycle_pos < 1.0 + half_active)).astype(np.float32)

            self.ampL_target = float(self._corridor_base_L * gate_L[-1])
            self.ampR_target = float(self._corridor_base_R * gate_R[-1])

            ampsL = self._corridor_base_L * gate_L
            ampsR = self._corridor_base_R * gate_R
        else:
            decay = (1.0 - self.smooth) ** np.arange(1, frames + 1)
            ampsL = self.ampL_target + (self.ampL_current - self.ampL_target) * decay
            ampsR = self.ampR_target + (self.ampR_current - self.ampR_target) * decay

        decay_b = (1.0 - self.smooth) ** np.arange(1, frames + 1)
        ampsB = self.beep_amp_target + (self.beep_amp_current - self.beep_amp_target) * decay_b

        self.ampL_current       = float(ampsL[-1])
        self.ampR_current       = float(ampsR[-1])
        self.beep_amp_current   = float(ampsB[-1])

        outdata[:] = 0.0
        beep = (ampsB * gate * tone).astype(np.float32)
        outdata[:, self.spkA] = beep
        outdata[:, self.spkB] = beep
        outdata[:, self.chL]  = (ampsL * sine).astype(np.float32)
        outdata[:, self.chR]  = (ampsR * sine).astype(np.float32)

    # ========================= lifecycle =========================
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
                time.sleep(glide_seconds)
            except Exception:
                pass
            self.stream.stop()
            self.stream.close()
            self.stream = None