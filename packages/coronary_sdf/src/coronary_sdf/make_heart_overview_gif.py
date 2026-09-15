"""Create a PowerPoint-friendly accelerated GIF from the HiP-CT stack video."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# Source video and output GIF: first CLI argument, else the environment, else
# a file beside the current directory. No machine-specific path is baked in.
SOURCE = Path(
    (sys.argv[1] if len(sys.argv) > 1 else "")
    or os.environ.get("CORONARY_SDF_VIDEO")
    or (Path.cwd() / "heart_overview.mp4")
)
OUTPUT = Path(
    (sys.argv[2] if len(sys.argv) > 2 else "")
    or (Path.cwd() / "heart_overview_01m40_to_02m20.gif")
)

OUTPUT_SIZE = 480
OUTPUT_FPS = 8
SPEEDUP = 1.0
GREY_LEVELS = 32
START_SECONDS = 100.0
END_SECONDS = 140.0


def grayscale_palette() -> list[int]:
    palette: list[int] = []
    for value in range(256):
        palette.extend((value, value, value))
    return palette


def main() -> None:
    capture = cv2.VideoCapture(str(SOURCE))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {SOURCE}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        raise RuntimeError("Video metadata is unavailable")

    start_frame = max(0, round(START_SECONDS * source_fps))
    end_frame = min(frame_count, round(END_SECONDS * source_fps))
    source_step = max(1, round(source_fps * SPEEDUP / OUTPUT_FPS))
    selected = set(range(start_frame, end_frame, source_step))
    frames: list[Image.Image] = []
    palette = grayscale_palette()
    quant_step = 256 // GREY_LEVELS

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    index = start_frame
    while True:
        if index >= end_frame:
            break
        ok, frame = capture.read()
        if not ok:
            break
        if index in selected:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey = cv2.resize(
                grey,
                (OUTPUT_SIZE, OUTPUT_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            # Fixed global greyscale palette prevents frame-to-frame palette flicker.
            quantised = ((grey.astype(np.uint16) + quant_step // 2) // quant_step)
            quantised = np.clip(quantised * quant_step, 0, 255).astype(np.uint8)
            image = Image.fromarray(quantised, mode="P")
            image.putpalette(palette)
            frames.append(image)
        index += 1

    capture.release()
    if not frames:
        raise RuntimeError("No GIF frames were generated")

    # Preserve the requested source-time playback even when the integer frame
    # stride does not divide the source frame rate exactly.
    ideal_duration_ms = 1000 * source_step / source_fps / SPEEDUP
    lower_ms = int(ideal_duration_ms // 10) * 10
    upper_ms = lower_ms + 10
    upper_fraction = (ideal_duration_ms - lower_ms) / 10
    frame_durations = []
    accumulator = 0.0
    for _ in frames:
        accumulator += upper_fraction
        if accumulator >= 1.0:
            frame_durations.append(upper_ms)
            accumulator -= 1.0
        else:
            frame_durations.append(lower_ms)
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=frame_durations,
        loop=0,
        disposal=1,
        optimize=True,
    )

    gif_duration = sum(frame_durations) / 1000
    effective_fps = len(frames) / gif_duration
    print(
        f"Wrote {OUTPUT} with {len(frames)} frames at {effective_fps:.2f} fps "
        f"({gif_duration:.1f} s, {SPEEDUP:g}x source speed)"
    )


if __name__ == "__main__":
    main()
