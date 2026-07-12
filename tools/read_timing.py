"""Capture the per-frame TIMING line from the STM32 streaming firmware.

The firmware (app_x-cube-ai.c streaming loop) prints over UART:

    AUDIO_START\r\n
    <1662 x 160 bytes raw float32 enhanced audio>
    TIMING:<ms> ms/frame RTF:<x>\r\n
    AUDIO_END\r\n

This script opens the COM port, waits for the ASCII "TIMING:" marker inside
the (mostly binary) stream, prints that line, and exits. Start it BEFORE
pressing the board's reset button so the start of the run is not missed.

Usage:  python tools/read_timing.py --port COM3 [--label conf_nobin_n64]
"""
from __future__ import annotations

import argparse
import sys
import time

import serial


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM3")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--label", default="", help="tag printed with the result")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="max seconds to wait for the TIMING line")
    args = p.parse_args()

    print(f"Opening {args.port} @ {args.baud} ... press the board RESET "
          "button now.")
    buf = b""
    t0 = time.time()
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        while time.time() - t0 < args.timeout:
            chunk = ser.read(4096)
            if not chunk:
                continue
            buf += chunk
            if b"AUDIO_START" in buf and len(buf) <= 4096:
                print("AUDIO_START seen, streaming ...")
            idx = buf.find(b"TIMING:")
            if idx >= 0:
                end = buf.find(b"\r\n", idx)
                while end < 0 and time.time() - t0 < args.timeout:
                    buf += ser.read(256)
                    end = buf.find(b"\r\n", idx)
                line = buf[idx:end].decode(errors="replace")
                tag = f"[{args.label}] " if args.label else ""
                print(f"{tag}{line}")
                return
    print("ERROR: no TIMING line received before timeout", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
