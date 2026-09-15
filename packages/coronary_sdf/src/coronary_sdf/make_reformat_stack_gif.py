"""Create a presentation-ready GIF from a HiP-CT reformat NPZ stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Return a one-pixel inner outline without requiring scipy."""
    m = mask.astype(bool)
    eroded = m.copy()
    eroded[1:, :] &= m[:-1, :]
    eroded[:-1, :] &= m[1:, :]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    return m & ~eroded


def fixed_palette() -> list[int]:
    """224 grey levels plus a fixed cyan segmentation colour."""
    palette: list[int] = []
    for value in range(224):
        grey = round(value * 255 / 223)
        palette.extend((grey, grey, grey))
    palette.extend((0, 220, 255))
    palette.extend((0, 0, 0) * (256 - 225))
    return palette


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="render the raw greyscale stack without the segmentation boundary",
    )
    parser.add_argument("--low-percentile", type=float, default=0.5)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    args = parser.parse_args()

    output = args.output or args.input.with_name(f"{args.input.stem}_stack.gif")
    if args.fps <= 0 or args.scale <= 0:
        parser.error("--fps and --scale must be positive")

    with np.load(args.input, allow_pickle=False) as archive:
        raw = np.asarray(archive["raw"])
        mask = np.asarray(archive["mask"]) if "mask" in archive else None

    if raw.ndim != 3:
        raise ValueError(f"Expected raw to have shape (frames, height, width), got {raw.shape}")
    if mask is not None and mask.shape != raw.shape:
        raise ValueError(f"mask shape {mask.shape} does not match raw shape {raw.shape}")

    lo, hi = np.percentile(raw, [args.low_percentile, args.high_percentile])
    if not hi > lo:
        raise ValueError(f"Invalid intensity window: {lo} to {hi}")

    indexed = np.clip((raw.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    indexed = np.rint(indexed * 223).astype(np.uint8)
    palette = fixed_palette()
    frames: list[Image.Image] = []

    for index in range(raw.shape[0]):
        frame_data = indexed[index].copy()
        if mask is not None and not args.no_overlay:
            frame_data[mask_outline(mask[index])] = 224
        frame = Image.fromarray(frame_data, mode="P")
        frame.putpalette(palette)
        if args.scale != 1:
            frame = frame.resize(
                (raw.shape[2] * args.scale, raw.shape[1] * args.scale),
                resample=Image.Resampling.NEAREST,
            )
            frame.putpalette(palette)
        frames.append(frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000 / args.fps))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    print(f"Wrote {output}")
    print(f"Frames: {len(frames)}")
    print(f"Dimensions: {frames[0].width} x {frames[0].height}")
    print(f"Playback: {args.fps:g} fps ({len(frames) / args.fps:.2f} s)")
    print(f"Intensity window: {lo:.1f} to {hi:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
