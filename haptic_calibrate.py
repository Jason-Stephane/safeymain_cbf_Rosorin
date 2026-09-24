#!/usr/bin/env python3
# =================================================================================================
# haptic_calibrate.py  —  Per-participant haptic calibration tool (no ROS needed).
#
# Workflow:
#   1. Run this script before the experiment.
#   2. Vibration ramps up from zero to maximum over ~15 seconds.
#   3. Participant presses [L] when they FIRST feel the vibration (perception threshold).
#   4. Participant presses [H] when vibration is the MAXIMUM they'd want (comfort ceiling).
#   5. Script saves both values to a JSON file keyed by participant ID.
#   6. haptic_node6.py loads this file and maps its 0-1 intensity into the calibrated range.
#
# The script also does a confirmation pass: after marking both thresholds, it plays
# the calibrated range back so the participant can verify it feels right.
#
# Usage:
#   python3 haptic_calibrate.py                                     # prompted for ID
#   python3 haptic_calibrate.py --participant P01                   # full calibration
#   python3 haptic_calibrate.py --participant P01 --replay          # replay saved values
#   python3 haptic_calibrate.py --participant P01 --master 0.15     # higher max amplitude
# =================================================================================================

import argparse
import json
import os
import sys
import time
import termios
import tty
import threading

# Route audio to the DualSense controller's haptic motors (same sink as haptic_node6)
os.environ.setdefault(
    "PULSE_SINK",
    "alsa_output.usb-Sony_Interactive_Entertainment_Wireless_Controller-00.analog-surround-40",
)

from haptic_core5 import HapticAudioEngine

# ---------------------------------------------------------------------------
# Keyboard input helper (non-blocking single keypress)
# ---------------------------------------------------------------------------
_latest_key = None
_stop_listener = False
_saved_termios = None   # saved BEFORE going raw, restored on any exit


def _save_terminal():
    """Save terminal state before anything touches it. Call once at startup."""
    global _saved_termios
    try:
        _saved_termios = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        _saved_termios = None


def _restore_terminal():
    """Restore terminal to its original state. Safe to call multiple times."""
    if _saved_termios is not None:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _saved_termios)
        except Exception:
            pass


def _key_listener():
    """Background thread: reads single keypresses (Unix terminal raw mode)."""
    global _latest_key, _stop_listener
    try:
        tty.setraw(sys.stdin.fileno())
        while not _stop_listener:
            ch = sys.stdin.read(1)
            if ch:
                _latest_key = ch.lower()
                if ch == '\x03':  # Ctrl-C
                    _stop_listener = True
    finally:
        _restore_terminal()


def get_key():
    """Return the latest keypress (or None), then clear it."""
    global _latest_key
    k = _latest_key
    _latest_key = None
    return k


# ---------------------------------------------------------------------------
# Ramp and calibration
# ---------------------------------------------------------------------------
def run_calibration(engine, ramp_duration=15.0, direction_deg=0.0):
    """
    Ramp vibration from 0 to master over ramp_duration seconds.
    Returns (master_low, master_high) — the raw amplitude values at the
    participant's perception threshold and comfort ceiling.
    """
    master = engine.master
    print("\n" + "=" * 60)
    print("  HAPTIC CALIBRATION")
    print("=" * 60)
    print(f"\n  Vibration will ramp from 0 to max over {ramp_duration:.0f} seconds.")
    print("  Press [L] when you FIRST feel the vibration.")
    print("  Press [H] when it reaches the STRONGEST you'd want.")
    print("  Press [Q] to abort.\n")
    print("  Starting in 3 seconds...", flush=True)
    time.sleep(3.0)

    master_low = None
    master_high = None
    t0 = time.monotonic()

    while True:
        elapsed = time.monotonic() - t0
        progress = min(1.0, elapsed / ramp_duration)
        intensity = progress  # linear ramp 0→1

        left, right = engine.directional(direction_deg)
        engine.ampL_target = master * intensity * left
        engine.ampR_target = master * intensity * right

        # Progress bar
        bar_len = 40
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        amp_now = master * intensity
        status_l = f"  L={master_low:.4f}" if master_low else "  L=---"
        status_h = f"  H={master_high:.4f}" if master_high else "  H=---"
        sys.stdout.write(
            f"\r  [{bar}] {progress*100:5.1f}%  amp={amp_now:.4f}"
            f"{status_l}{status_h}   "
        )
        sys.stdout.flush()

        key = get_key()
        if key == 'l' and master_low is None:
            master_low = amp_now
            print(f"\n  ✓ Perception threshold marked: {master_low:.4f}", flush=True)
        elif key == 'h':
            master_high = amp_now
            print(f"\n  ✓ Comfort ceiling marked: {master_high:.4f}", flush=True)
        elif key == 'q' or key == '\x03':
            print("\n  Aborted.", flush=True)
            return None, None

        # If both marked, we're done
        if master_low is not None and master_high is not None:
            break

        # If ramp finished without both marks
        if progress >= 1.0:
            if master_low is None:
                print("\n\n  ⚠ You didn't mark the perception threshold (L).", flush=True)
                print("  Try again with a higher --master value.", flush=True)
                return None, None
            if master_high is None:
                master_high = master  # default to max
                print(f"\n  (Comfort ceiling defaulted to max: {master_high:.4f})", flush=True)
            break

        time.sleep(0.03)

    # Silence
    engine.silence()
    time.sleep(0.3)

    # Ensure low < high
    if master_high <= master_low:
        master_high = min(master, master_low * 1.5)
        print(f"  (Adjusted ceiling to {master_high:.4f} — was <= threshold)", flush=True)

    return master_low, master_high


def run_confirmation(engine, master_low, master_high, duration=6.0, direction_deg=0.0):
    """
    Play back the calibrated range: ramp from master_low to master_high over duration.
    Participant confirms it feels right.
    """
    print("\n" + "-" * 60)
    print("  CONFIRMATION: playing your calibrated range...")
    print(f"  Low={master_low:.4f} → High={master_high:.4f}")
    print("  Press [Y] to accept, [N] to redo, [Q] to abort.")
    print("-" * 60, flush=True)
    time.sleep(1.0)

    t0 = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0
        progress = min(1.0, elapsed / duration)

        # Triangle wave: ramp up then back down
        if progress < 0.5:
            t = progress * 2.0
        else:
            t = (1.0 - progress) * 2.0
        amp = master_low + t * (master_high - master_low)

        left, right = engine.directional(direction_deg)
        engine.ampL_target = amp * left
        engine.ampR_target = amp * right

        bar_len = 30
        filled = int(bar_len * t)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] amp={amp:.4f}  ")
        sys.stdout.flush()

        key = get_key()
        if key == 'y':
            print("\n  ✓ Calibration accepted.", flush=True)
            engine.silence()
            return True
        elif key == 'n':
            print("\n  ↻ Will redo calibration.", flush=True)
            engine.silence()
            return False
        elif key == 'q' or key == '\x03':
            print("\n  Aborted.", flush=True)
            engine.silence()
            return None

        if progress >= 1.0:
            # Loop the confirmation
            t0 = time.monotonic()

        time.sleep(0.03)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Per-participant haptic calibration")
    parser.add_argument("--participant", "-p", default=None,
                        help="Participant ID (e.g. P01). Prompted interactively if omitted.")
    parser.add_argument("--output", "-o", default="haptic_calibration.json",
                        help="Output JSON file (default: haptic_calibration.json)")
    parser.add_argument("--replay", action="store_true",
                        help="Replay saved calibration for a participant instead of re-calibrating")
    parser.add_argument("--master", type=float, default=0.1,
                        help="Max haptic amplitude (default: 0.1)")
    parser.add_argument("--ramp-duration", type=float, default=15.0,
                        help="Ramp-up duration in seconds (default: 15)")
    parser.add_argument("--device", default="pulse", help="Audio device (default: pulse)")
    parser.add_argument("--freq", type=float, default=60.0,
                        help="Vibration frequency in Hz (default: 60)")
    args = parser.parse_args()

    global _stop_listener

    # Save terminal state FIRST — before anything can corrupt it
    _save_terminal()

    # Interactive prompt happens BEFORE raw mode (normal terminal input)
    pid = args.participant
    if pid is None:
        pid = input("  Enter participant ID: ").strip()
        if not pid:
            print("  No ID entered. Exiting.")
            return
    args.participant = pid

    engine = None
    try:
        engine = HapticAudioEngine(
            device=args.device,
            haptic_channels=(2, 3),
            freq=args.freq,
            master=args.master,
            smooth=0.0008,
        )
        engine.start()
    except Exception as e:
        print(f"\n  ERROR: Could not start haptic engine: {e}")
        print("  Check that the DualSense controller is connected and the")
        print("  PULSE_SINK matches your audio device.")
        print("  (Run `pactl list short sinks` to see available sinks)")
        _restore_terminal()
        return

    # Start keyboard listener AFTER engine is confirmed working
    listener = threading.Thread(target=_key_listener, daemon=True)
    listener.start()

    try:
        # ---- Replay mode: load saved values and play them back ----
        if args.replay:
            if not os.path.exists(args.output):
                print(f"\n  No calibration file found: {args.output}")
                return
            with open(args.output, 'r') as f:
                cal_data = json.load(f)
            if args.participant not in cal_data:
                print(f"\n  No calibration for '{args.participant}' in {args.output}")
                print(f"  Available IDs: {', '.join(cal_data.keys()) or '(none)'}")
                return
            entry = cal_data[args.participant]
            master_low = entry["master_low"]
            master_high = entry["master_high"]
            saved_master = entry.get("master_max", args.master)
            engine.master = saved_master
            print(f"\n  Replaying calibration for {args.participant}:")
            print(f"    low={master_low:.4f}  high={master_high:.4f}  master={saved_master:.4f}")

            while True:
                result = run_confirmation(engine, master_low, master_high)
                if result is True:
                    _restore_terminal()
                    print(f"\n  Calibration confirmed for {args.participant}.")
                    break
                elif result is False:
                    # Redo → switch to full calibration
                    _restore_terminal()
                    print("\n  Switching to full re-calibration...")
                    _save_terminal()
                    # Fall through to calibration loop below
                    args.replay = False
                    break
                else:
                    break

            if args.replay:
                # Confirmed or aborted — done
                return

        # ---- Full calibration loop ----
        while True:
            master_low, master_high = run_calibration(
                engine, ramp_duration=args.ramp_duration)

            if master_low is None:
                break

            result = run_confirmation(engine, master_low, master_high)
            if result is True:
                # Save
                cal_data = {}
                if os.path.exists(args.output):
                    with open(args.output, 'r') as f:
                        cal_data = json.load(f)

                cal_data[args.participant] = {
                    "master_low": round(master_low, 6),
                    "master_high": round(master_high, 6),
                    "master_max": args.master,
                    "freq": args.freq,
                }

                with open(args.output, 'w') as f:
                    json.dump(cal_data, f, indent=2)

                # Restore terminal before printing results (so output looks normal)
                _restore_terminal()
                print(f"\n  Saved to {args.output}:")
                print(f"    {args.participant}: low={master_low:.4f}  high={master_high:.4f}")
                print(f"\n  To use in haptic_node6.py:")
                print(f"    -p calibration_file:={os.path.abspath(args.output)} "
                      f"-p participant_id:={args.participant}")
                break
            elif result is None:
                break
            # result is False → loop back to redo

    except KeyboardInterrupt:
        pass
    finally:
        _stop_listener = True
        if engine is not None:
            engine.stop()
        _restore_terminal()
        print("\n  Done.\n")


if __name__ == "__main__":
    main()