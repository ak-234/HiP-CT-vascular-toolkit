"""Export the selected HiP-CT stack interval as a native-resolution GIF."""

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
    or (Path.cwd() / "heart_overview_01m40_to_02m20_fullres.gif")
)

START_SECONDS = 100.0
END_SECONDS = 140.0
OUTPUT_FPS = 10


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
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, round(START_SECONDS * source_fps))
    end_frame = min(frame_count, round(END_SECONDS * source_fps))
    source_step = max(1, round(source_fps / OUTPUT_FPS))

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    palette = grayscale_palette()
    frames: list[Image.Image] = []
    index = start_frame

    while index < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (index - start_frame) % source_step == 0:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(np.ascontiguousarray(grey), mode="P")
            image.putpalette(palette)
            frames.append(image)
        index += 1

    capture.release()
    if not frames:
        raise RuntimeError("No GIF frames were generated")

    duration_ms = round(1000 * source_step / source_fps)
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=1,
        optimize=False,
    )

    print(
        f"Wrote {OUTPUT}: {width}x{height}, {len(frames)} frames, "
        f"{len(frames) * duration_ms / 1000:.1f} s"
    )


if __name__ == "__main__":
    main()
