"""The single physical coordinate frame that ties all four inputs together.

Everything is reduced to **micrometres**, with the raw TIFF stack defining the grid:

    x_um = col   * voxel_x        (Amira 'uniform' convention: voxel 0 sits at 0)
    y_um = row   * voxel_y
    z_um = slice * voxel_z

The segmentation is a binned, cropped lattice positioned by its ``BoundingBox``; the
reconstructed surface is in millimetres. ``WorldFrame`` holds the mapping and
``validate`` proves it against the data rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MM_TO_UM = 1000.0


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    fatal: bool = False

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.fatal else "WARN")
        return f"  [{mark}] {self.name:<38s} {self.detail}"


class ValidationError(RuntimeError):
    pass


@dataclass
class WorldFrame:
    """Maps between raw voxel indices, segmentation voxel indices, and micrometres."""

    raw_shape: tuple  # (n_slices, n_rows, n_cols)
    raw_voxel: np.ndarray  # (3,) um, ordered (x, y, z)
    seg_dims: np.ndarray  # (3,) (nx, ny, nz)
    seg_origin: np.ndarray  # (3,) um, centre of segmentation voxel (0, 0, 0)
    seg_spacing: np.ndarray  # (3,) um
    stl_scale: float = MM_TO_UM
    nominal_voxel: np.ndarray = None  # (3,) um as supplied/detected, before refinement
    #: (3,) factor taking a length in the files' own micrometres to a true one. Ones
    #: unless the stated voxel size disagreed with the segmentation bounding box and
    #: was taken as the truth -- see :meth:`from_inputs`.
    world_scale: np.ndarray = None

    def __post_init__(self):
        if self.world_scale is None:
            self.world_scale = np.ones(3, dtype=np.float64)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_inputs(cls, raw_shape, raw_voxel, lattice, stl_scale=MM_TO_UM,
                    *, voxel_is_truth: bool = False) -> "WorldFrame":
        """Build the frame from the raw grid, the segmentation lattice and a voxel size.

        Two readings of the same three inputs, and which one is taken matters by more
        than a label:

        * ``voxel_is_truth=False`` -- **the lattice bounding box is the authority.** The
          supplied voxel size only needs to be right to within ~25%: it establishes the
          integer binning factor, after which the exact raw voxel size is taken as
          ``seg_spacing / bin``. That is what every caller did before there was a
          choice, and it is right whenever the bounding box is right.
        * ``voxel_is_truth=True`` -- **the supplied voxel size is the authority.** The
          bounding box is read as a claim about the same lattice made in the wrong
          units, so the whole micrometre world is rescaled by
          ``stated * bin / bbox_spacing``, recorded in :attr:`world_scale`. Voxel
          *indices* are untouched, which is what keeps the images, the mask and the
          graph aligned with each other; every *length* changes.

        The second exists because a bounding box written from a rounded voxel size is
        wrong in a way nothing downstream can notice. It is internally consistent, so
        every check passes and every radius, length and volume in the session is
        quietly off by one factor. On LADAF-2024-28 the recorded 32.99 um against the
        acquisition's 32.04 um is 2.96% -- larger than most of the corrections
        `radius-perimeter` exists to make.

        A caller holding the true voxel size therefore has to say so, and the loader
        asks for it rather than inferring one; see ``main.Session.load``.
        """
        nominal = np.broadcast_to(np.asarray(raw_voxel, dtype=np.float64), (3,)).copy()
        spacing = np.asarray(lattice.spacing, dtype=np.float64)
        origin = np.asarray(lattice.origin, dtype=np.float64)
        binf = np.maximum(np.round(spacing / nominal), 1).astype(np.int64)
        if voxel_is_truth:
            corrected = nominal * binf
            # Per axis, because the three bounding-box spacings differ in their last
            # digits and one shared factor would leave a sub-voxel skew behind.
            scale = corrected / spacing
            return cls(
                raw_shape=tuple(int(v) for v in raw_shape),
                raw_voxel=nominal,
                seg_dims=np.asarray(lattice.dims, dtype=np.int64),
                # The origin is a length too, so it moves with everything else --
                # otherwise the lattice keeps its old offset at a new spacing.
                seg_origin=origin * scale,
                seg_spacing=corrected,
                stl_scale=float(stl_scale) * float(scale.mean()),
                nominal_voxel=nominal,
                world_scale=scale,
            )
        return cls(
            raw_shape=tuple(int(v) for v in raw_shape),
            raw_voxel=spacing / binf,
            seg_dims=np.asarray(lattice.dims, dtype=np.int64),
            seg_origin=origin,
            seg_spacing=spacing,
            stl_scale=float(stl_scale),
            nominal_voxel=nominal,
        )

    # -- the unit correction ----------------------------------------------- #
    @property
    def corrected(self) -> bool:
        """Whether the files' micrometres are being rescaled to reach true ones."""
        return bool(np.any(np.abs(np.asarray(self.world_scale) - 1.0) > 1e-9))

    @property
    def voxel_um(self) -> float:
        """The raw voxel size this frame works in, as one number."""
        return float(np.mean(self.raw_voxel))

    @property
    def file_voxel_um(self) -> float:
        """What the segmentation bounding box implies the raw voxel size is."""
        return float(np.mean(self.seg_spacing / np.maximum(self.bin_factor, 1)
                             / np.asarray(self.world_scale)))

    def scale_points(self, xyz) -> np.ndarray:
        """(N, 3) coordinates in the files' um -> the same points in true um."""
        return np.asarray(xyz, dtype=np.float64) * self.world_scale

    def scale_lengths(self, values) -> np.ndarray:
        """Radii and other isotropic lengths, which have no axis to be scaled along.

        The mean of the three factors. They differ only in their sixth digit -- a
        bounding box is written from one voxel size, and its axes disagree by rounding
        alone -- so any choice among them is the same number to five places; the mean
        is used so that no axis is silently privileged.
        """
        return np.asarray(values, dtype=np.float64) * float(np.mean(self.world_scale))

    def correction_note(self) -> str:
        """One line for the load log, or empty when nothing was corrected."""
        if not self.corrected:
            return ""
        k = float(np.mean(self.world_scale))
        return (
            f"the segmentation bounding box implies {self.file_voxel_um:.4f} um, you "
            f"gave {self.voxel_um:.4f} um. Every length in this session -- radii, "
            f"distances, and the coordinates written back out -- is scaled by "
            f"{k:.6f} ({100 * (k - 1):+.2f}%). Voxel indices are unchanged."
        )

    # -- derived binning --------------------------------------------------- #
    @property
    def bin_factor(self) -> np.ndarray:
        """(3,) integer binning of the segmentation relative to the raw grid."""
        return np.round(self.seg_spacing / self.raw_voxel).astype(np.int64)

    @property
    def raw_start(self) -> np.ndarray:
        """(3,) first raw index covered by segmentation voxel 0, per axis."""
        b = self.bin_factor
        return np.round(self.seg_origin / self.raw_voxel - (b - 1) / 2.0).astype(np.int64)

    @property
    def raw_extent_um(self) -> np.ndarray:
        nz, ny, nx = self.raw_shape
        return np.array([nx - 1, ny - 1, nz - 1], dtype=np.float64) * self.raw_voxel

    @property
    def seg_bbox_um(self) -> np.ndarray:
        """(6,) xmin xmax ymin ymax zmin zmax of the segmentation voxel centres."""
        hi = self.seg_origin + (self.seg_dims - 1) * self.seg_spacing
        return np.array(
            [self.seg_origin[0], hi[0], self.seg_origin[1], hi[1], self.seg_origin[2], hi[2]]
        )

    # -- conversions ------------------------------------------------------- #
    def raw_to_um(self, zyx) -> np.ndarray:
        """(N,3) raw (slice, row, col) -> (N,3) um (x, y, z)."""
        zyx = np.atleast_2d(np.asarray(zyx, dtype=np.float64))
        xyz = zyx[:, ::-1]  # (col, row, slice)
        return xyz * self.raw_voxel

    def um_to_raw(self, xyz) -> np.ndarray:
        """(N,3) um (x, y, z) -> (N,3) fractional raw (slice, row, col)."""
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        return (xyz / self.raw_voxel)[:, ::-1]

    def um_to_raw_index(self, xyz) -> np.ndarray:
        return np.round(self.um_to_raw(xyz)).astype(np.int64)

    def um_to_seg(self, xyz) -> np.ndarray:
        """(N,3) um -> (N,3) fractional segmentation (i, j, k) i.e. (x, y, z) order."""
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        return (xyz - self.seg_origin) / self.seg_spacing

    def um_to_seg_index(self, xyz) -> np.ndarray:
        return np.round(self.um_to_seg(xyz)).astype(np.int64)

    def seg_to_um(self, ijk) -> np.ndarray:
        ijk = np.atleast_2d(np.asarray(ijk, dtype=np.float64))
        return self.seg_origin + ijk * self.seg_spacing

    def raw_axis_to_seg_axis(self, idx, axis: int) -> np.ndarray:
        """Integer raw index -> segmentation index along one axis (0=x, 1=y, 2=z)."""
        idx = np.asarray(idx, dtype=np.int64)
        return (idx - self.raw_start[axis]) // self.bin_factor[axis]

    def raw_slice_to_seg_slice(self, slice_idx) -> np.ndarray:
        return self.raw_axis_to_seg_axis(slice_idx, 2)

    def stl_to_um(self, xyz) -> np.ndarray:
        return np.asarray(xyz, dtype=np.float64) * self.stl_scale

    def describe(self) -> str:
        nz, ny, nx = self.raw_shape
        b = self.bin_factor
        s = self.raw_start
        return (
            f"raw stack      {nx} x {ny} cols/rows x {nz} slices, voxel "
            f"{self.raw_voxel[0]:.6f} / {self.raw_voxel[1]:.6f} / {self.raw_voxel[2]:.6f} um\n"
            f"segmentation   {self.seg_dims[0]} x {self.seg_dims[1]} x {self.seg_dims[2]}, "
            f"spacing {self.seg_spacing[0]:.4f} / {self.seg_spacing[1]:.4f} / {self.seg_spacing[2]:.4f} um\n"
            f"               binned {b[0]}x{b[1]}x{b[2]}, cropped from raw index "
            f"(col {s[0]}, row {s[1]}, slice {s[2]})\n"
            f"surface scale  x{self.stl_scale:g} (mm -> um)"
            + (f"\nvoxel size     {self.correction_note()}" if self.corrected else "")
        )


# --------------------------------------------------------------------------- #
# Applying the correction to what the files hold
# --------------------------------------------------------------------------- #
#: Per-point and per-edge fields that are *lengths* and therefore move with the
#: correction. Everything else a graph carries -- an order, a tree id, a reason code,
#: a ratio -- is left alone, because scaling it would be nonsense rather than a
#: refinement. A field this list does not name keeps whatever units it was written in.
LENGTH_FIELDS = ("thickness", "radius", "Radius", "MeanRadius", "meanradius")


def rescale_graph(graph, frame) -> bool:
    """Put a graph read from file into the frame's micrometres. In place.

    Returns whether anything moved, so the caller can report it. A frame that is not
    correcting anything leaves the graph untouched and returns False, which is the
    ordinary case and costs one comparison.

    Coordinates scale per axis and radii scale by the mean, for the reason
    :meth:`WorldFrame.scale_lengths` gives. ``MeanRadius`` is included because
    `radius-perimeter` re-derives it: a graph whose coordinates were corrected and
    whose stored radii were not is worse than one that was never corrected at all,
    since nothing about it looks wrong.
    """
    if not frame.corrected:
        return False
    graph.points = frame.scale_points(graph.points)
    graph.vertices = frame.scale_points(graph.vertices)
    graph.thickness = frame.scale_lengths(graph.thickness)
    for attrs in (graph.point_attrs, graph.edge_attrs, graph.vertex_attrs):
        for name in list(attrs):
            if name in LENGTH_FIELDS:
                attrs[name] = frame.scale_lengths(attrs[name])
    return True


def rescale_triple(triple, frame) -> bool:
    """The same correction for the editable representation. In place.

    ``Triple`` holds points as ``{id: (x, y, z, radius)}`` and nodes as
    ``{id: (x, y, z, degree)}``, so the two are rewritten rather than multiplied.
    """
    if not frame.corrected:
        return False
    k = float(np.mean(frame.world_scale))
    sx, sy, sz = (float(v) for v in np.asarray(frame.world_scale, dtype=np.float64))
    for pid, (x, y, z, r) in list(triple.points.items()):
        triple.points[pid] = (x * sx, y * sy, z * sz, r * k)
    for nid, value in list(triple.nodes.items()):
        x, y, z = value[0] * sx, value[1] * sy, value[2] * sz
        triple.nodes[nid] = (x, y, z, *value[3:])
    return True


def rescale_mesh(mesh, frame):
    """The surface, already in micrometres, moved onto the corrected scale.

    Returns the mesh so a caller can write ``mesh = rescale_mesh(mesh, frame)``
    whether or not anything happened. ``stl_to_um`` already carries the correction for
    a mesh being read *now*; this is for one that was scaled before the frame existed.
    """
    if mesh is None or not frame.corrected:
        return mesh
    mesh.points = frame.scale_points(mesh.points)
    return mesh


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _contains(outer: np.ndarray, inner: np.ndarray, tol: float) -> bool:
    return bool(
        np.all(inner[0::2] >= outer[0::2] - tol) and np.all(inner[1::2] <= outer[1::2] + tol)
    )


def validate(
    frame: WorldFrame,
    graph,
    labels=None,
    mesh_um=None,
    n_sample: int = 20000,
    rng_seed: int = 0,
) -> list[Check]:
    """Prove that all four inputs really do share one coordinate space.

    ``labels`` is a ``rle.ByteRLELattice`` for the binary mask; ``mesh_um`` is a
    PyVista mesh already scaled into micrometres. Both are optional so that partial
    validation still works.
    """
    checks: list[Check] = []
    rng = np.random.default_rng(rng_seed)

    # 1. The two readings of the voxel size -- what you said it is, and what the
    #    segmentation bounding box implies -- either agree, or the difference has been
    #    taken as a unit error and corrected. Both are reported, never merged: the
    #    failure this row exists to catch is a bounding box that is internally
    #    consistent and physically wrong, which nothing else can see.
    b = frame.bin_factor
    nominal = frame.nominal_voxel if frame.nominal_voxel is not None else frame.raw_voxel
    if frame.corrected:
        k = float(np.mean(frame.world_scale))
        checks.append(
            Check(
                "voxel size corrected to yours",
                True,
                f"stated {frame.voxel_um:.4f} um; bounding box implied "
                f"{frame.file_voxel_um:.4f} um; every length scaled by {k:.6f} "
                f"({100 * (k - 1):+.2f}%) at bin {b[0]}x{b[1]}x{b[2]}",
            )
        )
    else:
        rel = np.abs(frame.raw_voxel - nominal) / nominal
        checks.append(
            Check(
                "raw voxel size consistent",
                rel.max() < 0.02,
                f"nominal {nominal[0]:.4f} um vs seg-derived "
                f"{frame.raw_voxel[0]:.6f}/{frame.raw_voxel[1]:.6f}/{frame.raw_voxel[2]:.6f} um "
                f"at bin {b[0]}x{b[1]}x{b[2]} (max {100 * rel.max():.3f}% off)",
                fatal=True,
            )
        )

    # 2. The crop offset lands exactly on the raw grid.
    off = frame.seg_origin / frame.raw_voxel - (b - 1) / 2.0
    off_err = np.abs(off - np.round(off)).max()
    s = frame.raw_start
    checks.append(
        Check(
            "crop offset on raw grid",
            off_err < 1e-2,
            f"raw start (col {s[0]}, row {s[1]}, slice {s[2]}), max err {off_err:.2e} voxel",
            fatal=True,
        )
    )

    # 3. The segmentation lattice fits inside the raw stack.
    raw_hi = frame.raw_extent_um
    seg = frame.seg_bbox_um
    inside_raw = bool(np.all(seg[0::2] >= -1e-6) and np.all(seg[1::2] <= raw_hi + 1e-6))
    checks.append(
        Check(
            "segmentation within raw stack",
            inside_raw,
            f"seg z {seg[4]:.0f}-{seg[5]:.0f} um vs raw 0-{raw_hi[2]:.0f} um "
            f"(slices {frame.raw_start[2]}-{frame.raw_start[2] + b[2] * frame.seg_dims[2] - 1})",
            fatal=True,
        )
    )

    # 4. Graph lies inside the segmentation it was derived from.
    gmin, gmax = graph.points.min(0), graph.points.max(0)
    gbox = np.array([gmin[0], gmax[0], gmin[1], gmax[1], gmin[2], gmax[2]])
    tol = frame.seg_spacing.max()
    checks.append(
        Check(
            "graph within segmentation bbox",
            _contains(seg, gbox, tol),
            f"graph x {gmin[0]:.0f}-{gmax[0]:.0f}, y {gmin[1]:.0f}-{gmax[1]:.0f}, "
            f"z {gmin[2]:.0f}-{gmax[2]:.0f} um",
            fatal=True,
        )
    )

    # 5+6. Surface scale and radius agreement.
    if mesh_um is not None:
        mb = np.asarray(mesh_um.bounds, dtype=np.float64)
        checks.append(
            Check(
                "surface within segmentation bbox",
                _contains(seg, mb, tol),
                f"surface x {mb[0]:.0f}-{mb[1]:.0f}, y {mb[2]:.0f}-{mb[3]:.0f}, "
                f"z {mb[4]:.0f}-{mb[5]:.0f} um (scale x{frame.stl_scale:g})",
                fatal=True,
            )
        )
        from scipy.spatial import cKDTree

        verts = np.asarray(mesh_um.points, dtype=np.float64)
        if len(verts) > n_sample:
            verts = verts[rng.choice(len(verts), n_sample, replace=False)]
        tree = cKDTree(graph.points)
        dist, idx = tree.query(verts)
        ratio_r = np.median(dist / np.maximum(graph.thickness[idx], 1e-9))
        checks.append(
            Check(
                "surface radius matches thickness",
                0.85 <= ratio_r <= 1.15,
                f"median |surface-to-centreline| / local thickness = {ratio_r:.3f} "
                f"(expect ~1.0; wrong unit scaling gives ~{frame.stl_scale:.0f})",
            )
        )

    # 7. The decisive test: centreline points must fall inside the mask, and every
    #    flipped interpretation must not.
    if labels is not None:
        pts = graph.points
        if len(pts) > n_sample:
            pts = pts[rng.choice(len(pts), n_sample, replace=False)]
        ijk = frame.um_to_seg_index(pts)
        nx, ny, nz = (int(v) for v in frame.seg_dims)
        keep = np.all((ijk >= 0) & (ijk < [nx, ny, nz]), axis=1)
        ijk = ijk[keep]

        variants = {
            "direct": ijk.copy(),
            "y-flip": np.column_stack([ijk[:, 0], ny - 1 - ijk[:, 1], ijk[:, 2]]),
            "x-flip": np.column_stack([nx - 1 - ijk[:, 0], ijk[:, 1], ijk[:, 2]]),
            "z-flip": np.column_stack([ijk[:, 0], ijk[:, 1], nz - 1 - ijk[:, 2]]),
            "xy-swap": np.column_stack([ijk[:, 1], ijk[:, 0], ijk[:, 2]]),
        }
        # Decode each needed z slice once.
        fracs = {}
        for name, v in variants.items():
            ok = np.all((v >= 0) & (v < [nx, ny, nz]), axis=1)
            v = v[ok]
            hits = 0
            order = np.argsort(v[:, 2])
            v = v[order]
            zs = v[:, 2]
            for z in np.unique(zs):
                sl = labels.slice_z(int(z))
                sub = v[zs == z]
                hits += int((sl[sub[:, 1], sub[:, 0]] > 0).sum())
            fracs[name] = 100.0 * hits / max(len(ijk), 1)

        direct = fracs["direct"]
        worst_flip = max(v for k, v in fracs.items() if k != "direct")
        checks.append(
            Check(
                "centreline inside mask",
                direct >= 95.0,
                f"{direct:.1f}% of {len(ijk)} points (expect ~100%)",
                fatal=True,
            )
        )
        checks.append(
            Check(
                "orientation unambiguous",
                worst_flip < 20.0 and direct > 4 * worst_flip,
                "flips: " + ", ".join(f"{k} {v:.1f}%" for k, v in fracs.items() if k != "direct"),
                fatal=True,
            )
        )

    return checks


def report(checks: list[Check], strict: bool = True) -> None:
    """Print the validation table and raise if a fatal check failed."""
    print("Coordinate-space validation")
    for c in checks:
        print(c)
    bad = [c for c in checks if not c.passed and c.fatal]
    warn = [c for c in checks if not c.passed and not c.fatal]
    if warn:
        print(f"  ({len(warn)} non-fatal warning(s))")
    if bad and strict:
        raise ValidationError(
            "inputs are not in a common coordinate space:\n"
            + "\n".join(f"  - {c.name}: {c.detail}" for c in bad)
        )
    if not bad:
        print("  All structural checks passed.\n")
