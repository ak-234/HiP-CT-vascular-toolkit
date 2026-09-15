"""Lazy access to a directory of single-page image slices.

HiP-CT stacks run to thousands of slices and tens of GB, so nothing is loaded until
a specific slice (and usually only a window of it) is asked for.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile

# e.g. "32.99um_...", "32_99um_...", "6.5 um"
_VOXEL_RE = re.compile(r"(\d+(?:[._]\d+)?)\s*um", re.IGNORECASE)


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def infer_voxel_um(directory: Path, first_file: Path) -> float | None:
    """Best-effort nominal voxel size, from the folder/file name or a resolution tag.

    Only needs to be approximately right -- `WorldFrame.from_inputs` refines it against
    the segmentation lattice spacing.
    """
    for text in (directory.name, first_file.name, str(directory)):
        m = _VOXEL_RE.search(text)
        if m:
            return float(m.group(1).replace("_", "."))
    # ResolutionUnit is TIFF-specific.  Other formats still get the much more common
    # directory/file-name inference above, or can use ``--voxel-um``.
    if first_file.suffix.lower() not in TiffStack.TIFF_SUFFIXES:
        return None
    try:
        with tifffile.TiffFile(first_file) as tf:
            tags = tf.pages[0].tags
            if "XResolution" in tags and "ResolutionUnit" in tags:
                num, den = tags["XResolution"].value
                if num:
                    per_unit = den / num
                    unit = int(tags["ResolutionUnit"].value)
                    if unit == 3:  # centimetre
                        return per_unit * 1e4
                    if unit == 2:  # inch
                        return per_unit * 25400.0
    except Exception:
        pass
    return None


class TiffStack:
    """A z-ordered directory of 2D image slices with a small slice cache.

    The historical name is retained because it is part of the internal API.  TIFF is
    decoded with :mod:`tifffile`; JPEG 2000 and other conventional image files are
    decoded with Pillow.
    """

    #: Only these are slices.  Keeping an explicit allow-list is important: ESRF
    #: reconstruction directories often contain files such as ``.tif.dat`` and
    #: ``.tif.fcp`` beside slice zero, and counting those silently shifts every z index.
    TIFF_SUFFIXES = (".tif", ".tiff")
    PIL_SUFFIXES = (
        ".jp2",
        ".j2k",
        ".j2c",
        ".jpc",  # JPEG 2000
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    )
    IMAGE_SUFFIXES = TIFF_SUFFIXES + PIL_SUFFIXES

    #: Above this fraction of the page's rows, decoding strip by strip stops paying and
    #: :meth:`read_window` uses the whole-page path. Measured on LADAF-2024-28: a
    #: 130-row window is 18.8x faster than the page, 400 rows is 8.3x, and the two meet
    #: somewhere around two thirds of the 3079 rows.
    STRIP_WINDOW_MAX_FRACTION = 0.65

    def __init__(self, directory: str | Path, pattern: str = "*", cache_slices: int = 24):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise NotADirectoryError(f"raw stack directory not found: {self.directory}")

        # Filter by suffix, not by the glob alone. `*.tif*` matches `.tif.dat`,
        # `.tif.fcp`, `.tif.lda` and their `.bck` twins -- the sidecars ESRF's
        # reconstruction writes beside slice zero. On the LADAF-2024-28 overview that is
        # five extra files, and because they sort immediately after
        # `..._000000.tif` they take indices 1-5 and push **every real slice down by
        # five**. Nothing downstream can notice: `shape` merely reports 4757 instead of
        # 4752, `WorldFrame` derives its binning from that wrong number, and every raw
        # sample is then read from the wrong slice. Silent, and wrong everywhere `--raw`
        # is used.
        matched = sorted(self.directory.glob(pattern), key=natural_key)
        self.files = [
            p for p in matched if p.is_file() and p.suffix.lower() in self.IMAGE_SUFFIXES
        ]
        #: How many the suffix filter removed, so a caller can say so.
        self.skipped = len(matched) - len(self.files)
        if not self.files:
            raise FileNotFoundError(
                f"no supported image files matching '{pattern}' in {self.directory}; "
                f"supported suffixes: {', '.join(self.IMAGE_SUFFIXES)}"
                + (f" ({self.skipped} other file(s) ignored)" if self.skipped else "")
            )

        shape, self.dtype = self._image_info(self.files[0])
        self.n_rows, self.n_cols = shape
        self.n_slices = len(self.files)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = int(cache_slices)
        self.nominal_voxel_um = infer_voxel_um(self.directory, self.files[0])

        #: None until tried, True once a strip-level window has worked, False once one
        #: has failed for a structural reason (tiled, not a TIFF, decoder surprise).
        self._strip_reads_ok: bool | None = None
        #: How `read_window` got its pixels. A run that quietly fell back to whole pages
        #: is otherwise invisible except as an unexplained order of magnitude.
        self.strip_reads = 0
        self.whole_page_reads = 0

    @classmethod
    def _image_info(cls, path: Path) -> tuple[tuple[int, int], np.dtype]:
        """Return plane shape and dtype without decoding pixels."""
        if path.suffix.lower() in cls.TIFF_SUFFIXES:
            with tifffile.TiffFile(path) as tf:
                page = tf.pages[0]
                shape = tuple(int(v) for v in page.shape)
                if len(shape) != 2:
                    raise ValueError(f"{path.name} is not a 2D slice (shape {shape})")
                return (shape[0], shape[1]), np.dtype(page.dtype)
        try:
            from PIL import Image, ImageMode
        except ImportError as exc:
            raise RuntimeError(
                f"reading {path.suffix} slices requires Pillow; install the 'pillow' package"
            ) from exc
        try:
            with Image.open(path) as source:
                if len(source.getbands()) != 1:
                    raise ValueError(
                        f"{path.name} is not a 2D grayscale slice "
                        f"(mode {source.mode}, bands {source.getbands()})"
                    )
                dtype = np.dtype(ImageMode.getmode(source.mode).typestr)
                return (int(source.height), int(source.width)), dtype
        except ValueError:
            raise
        except Exception as exc:
            if path.suffix.lower() in (".jp2", ".j2k", ".j2c", ".jpc"):
                raise RuntimeError(
                    f"could not read JPEG 2000 metadata from {path.name}; install a Pillow "
                    "build with JPEG 2000 (OpenJPEG) support"
                ) from exc
            raise

    @classmethod
    def _read_image(cls, path: Path) -> np.ndarray:
        """Read one scalar plane, choosing a decoder from its suffix."""
        if path.suffix.lower() in cls.TIFF_SUFFIXES:
            image = tifffile.imread(path)
        else:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError(
                    f"reading {path.suffix} slices requires Pillow; install the 'pillow' package"
                ) from exc
            try:
                with Image.open(path) as source:
                    image = np.array(source)
            except Exception as exc:
                if path.suffix.lower() in (".jp2", ".j2k", ".j2c", ".jpc"):
                    raise RuntimeError(
                        f"could not decode JPEG 2000 slice {path.name}; install a Pillow build "
                        "with JPEG 2000 (OpenJPEG) support"
                    ) from exc
                raise

        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError(f"{path.name} is not a 2D grayscale slice (shape {image.shape})")
        return image

    @property
    def shape(self) -> tuple:
        """(n_slices, n_rows, n_cols)."""
        return (self.n_slices, self.n_rows, self.n_cols)

    def read_slice(self, z: int) -> np.ndarray:
        if not 0 <= z < self.n_slices:
            raise IndexError(f"slice {z} out of range [0, {self.n_slices})")
        hit = self._cache.get(z)
        if hit is not None:
            self._cache.move_to_end(z)
            return hit
        img = self._read_image(self.files[z])
        if img.shape != (self.n_rows, self.n_cols):
            raise ValueError(
                f"{self.files[z].name} has shape {img.shape}; expected "
                f"{(self.n_rows, self.n_cols)}"
            )
        self._cache[z] = img
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img

    def read_window(self, z: int, row0: int, row1: int, col0: int, col1: int) -> np.ndarray:
        """Sub-region of one slice, zero-padded where it runs off the edge.

        Decodes only the strips the window needs, where the file allows it. A striped
        TIFF compresses each strip independently, so a window a few hundred rows tall
        does not need the other few thousand -- and the whole-page decode this used to
        do is the single largest cost in an oblique reformat. Measured on the
        LADAF-2024-28 overview (3079 rows, ``rowsperstrip`` 1, LZW): 188 ms for the page
        against 10 ms for a 130-row window, **18.8x**, bit-identical.

        Falls back to the whole-page path whenever that is not available or not worth
        it; see :meth:`_strip_window`.
        """
        h, w = row1 - row0, col1 - col0
        out = np.zeros((h, w), dtype=self.dtype)
        r0, r1 = max(0, row0), min(self.n_rows, row1)
        c0, c1 = max(0, col0), min(self.n_cols, col1)
        if r0 >= r1 or c0 >= c1:
            return out

        # The cache first: a plane already decoded is strictly better than decoding
        # any part of it again.
        plane = self._cache.get(z)
        if plane is not None:
            self._cache.move_to_end(z)
            block = plane[r0:r1, c0:c1]
        else:
            if not 0 <= z < self.n_slices:
                raise IndexError(f"slice {z} out of range [0, {self.n_slices})")
            block = self._strip_window(self.files[z], r0, r1, c0, c1)
            if block is None:
                block = self.read_slice(z)[r0:r1, c0:c1]
                self.whole_page_reads += 1
            else:
                self.strip_reads += 1
        out[r0 - row0 : r1 - row0, c0 - col0 : c1 - col0] = block
        return out

    def _strip_window(self, path: Path, r0: int, r1: int, c0: int, c1: int):
        """Decode only the strips covering rows ``[r0, r1)``. ``None`` if not worth it.

        Returns ``None`` -- meaning "use the whole-page path" -- rather than raising,
        for every case where strip access does not apply or does not pay:

        * not a TIFF (the Pillow formats have no strip concept here);
        * a **tiled** page, where the segments are 2D tiles and the row arithmetic below
          does not hold;
        * a page whose strips are so tall that the window spans most of them anyway;
        * a window covering most of the rows, where the per-strip overhead outweighs
          what is skipped. On the measured file the crossover is around two thirds of
          the page: 400 rows still runs 8.3x faster, 3079 rows would not.

        The first failure for structural reasons is remembered on the stack, so a
        directory that cannot do this pays the check once rather than per slice.
        """
        if self._strip_reads_ok is False:
            return None
        if path.suffix.lower() not in self.TIFF_SUFFIXES:
            self._strip_reads_ok = False
            return None
        if (r1 - r0) > self.STRIP_WINDOW_MAX_FRACTION * self.n_rows:
            return None

        try:
            with tifffile.TiffFile(path) as tf:
                page = tf.pages[0]
                per = int(page.rowsperstrip or 0)
                if page.is_tiled or per <= 0:
                    self._strip_reads_ok = False
                    return None
                # A strip taller than the window itself saves nothing.
                if per > (r1 - r0):
                    return None

                handle = tf.filehandle
                s0, s1 = r0 // per, -(-r1 // per)  # ceil for the upper strip
                rows = np.zeros(((s1 - s0) * per, c1 - c0), dtype=self.dtype)
                for s in range(s0, s1):
                    count = int(page.databytecounts[s])
                    if count <= 0:
                        continue  # a sparse strip is legitimately zeros
                    handle.seek(int(page.dataoffsets[s]))
                    segment, _index, _shape = page.decode(handle.read(count), s)
                    # Use the shape the decoder reports rather than assuming one row
                    # per strip: `rowsperstrip` is a property of the file, not of TIFF.
                    flat = np.asarray(segment).reshape(-1, int(page.imagewidth))
                    at = (s - s0) * per
                    rows[at : at + len(flat)] = flat[:, c0:c1]
                self._strip_reads_ok = True
                return rows[r0 - s0 * per : r1 - s0 * per]
        except Exception:  # noqa: BLE001 - any decoder surprise falls back, never fails
            self._strip_reads_ok = False
            return None

    def read_stack_window(self, z_lo: int, z_hi: int, row0, row1, col0, col1) -> np.ndarray:
        """(z_hi-z_lo, h, w) window across a contiguous slice range."""
        return np.stack(
            [self.read_window(z, row0, row1, col0, col1) for z in range(z_lo, z_hi)], axis=0
        )
