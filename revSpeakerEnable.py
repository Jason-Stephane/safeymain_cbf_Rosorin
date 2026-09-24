#!/usr/bin/python3
"""Enable DualSense internal speaker + set volume. Run once after USB plug-in."""
import glob, os

dev = None
for uevent in glob.glob('/sys/class/hidraw/hidraw*/device/uevent'):
    if '054C' in open(uevent).read().upper() and ('0CE6' in open(uevent).read().upper()):
        dev = '/dev/' + uevent.split('/')[4]
        break
assert dev, "DualSense hidraw not found"
print("using", dev)

report = bytearray(48)
report[0] = 0x02          # USB output report ID
report[1] = 0x20 | 0x80   # valid_flag0: speaker-volume-enable | audio-control-enable
report[6] = 0xFF          # speaker volume, max
report[8] = 0x20          # audio control: output path bits -> speaker gets a channel
                          # (if silent, try 0x30 = speaker-only path)

with open(dev, 'wb') as f:
    f.write(bytes(report))
print("speaker enabled")