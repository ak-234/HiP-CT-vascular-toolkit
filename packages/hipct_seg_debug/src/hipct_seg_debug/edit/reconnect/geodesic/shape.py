"""Complete the mask by carrying the vessel's own cross-section across the gap.

The obvious way to fill a mask gap is to sweep a sphere or a tapered capsule along
the route, and :func:`~..segmentation.paint_bridges` does exactly that. It is
right for a centreline bridge and wrong here, for a reason that is easy to miss:

**this is an ex-vivo specimen, and its vessels are collapsed.** A distal coronary
in an unperfused HiP-CT block is not a circular tube. It is a slit, a ribbon, a
flattened ellipse -- and the segmentation faithfully records that. Painting a
circular capsule across a gap therefore does not repair the vessel; it *inflates*
it, inserting a few hundred micrometres of anatomically impossible round lumen
between two flattened ends. Every measurement downstream then reads a calibre step
that the specimen does not have, and the surface reconstruction meshes a bulge.

So the shape is transported rather than assumed:

1. Cut the plane perpendicular to the vessel at each intact end, and keep the
   2D blob belonging to *this* component -- not everything in the plane, which
   would pick up whichever neighbour the cut happened to graze.
2. Convert each to a 2D signed distance field. Interpolating occupancy directly
   produces the classic morphing artefact where a slit fades out and a second one
   fades in; interpolating signed distance rotates and shears the shape into the
   other one, which is what a vessel actually does along its length.
3. Carry them along the route on a **rotation-minimizing frame**. A Frenet frame
   is unusable here -- it flips through every inflection of the centreline and
   spins without bound where the curvature vanishes, which is most of a nearly
   straight bridge -- so the double-reflection method is used instead, which
   transports the normal with no twist at all.
4. Rasterise the interpolated shape, and add the minimum 26-connected core needed
   to guarantee continuity. Nothing more: where the transported shape already
   connects, no extra voxel is invented.

The radius written on the *graph* is a separate question and is kept separate. The
mask records what was observed; the graph carries a perimeter-equivalent radius,
which is the number the surface pipeline and every morphometric downstream expect.
Conflating the two is what re-inflates a collapsed specimen by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Cross-section grid half-width, in local radii.
SECTION_HALF_RADII = 3.0
#: ...and a floor in voxels, so a sub-voxel twig still gets a usable window.
SECTION_HALF_MIN = 5
#: A transported section thinner than this many voxels cannot be rasterised
#: reliably; the minimum core carries continuity there instead.
MIN_SECTION_VOXELS = 1


@dataclass
class Section:
    """One cross-section of a vessel, as a signed distance field on a local grid.

    ``sdf`` is negative inside the lumen and is in **micrometres**, so two
    sections cut at different local scales remain comparable and interpolating
    between them is meaningful.
    """

    sdf: np.ndarray  # (2h+1, 2h+1) float32, um; negative inside
    u: np.ndarray  # unit in-plane axis, world (x, y, z)
    v: np.ndarray
    tangent: np.ndarray
    centre_um: np.ndarray
    step_um: float  # grid spacing of the (u, v) sampling
    area_um2: float
    perimeter_um: float
    valid: bool = True
    reason: str = ""

    @property
    def r_area(self) -> float:
        """Area-equivalent radius: what the observed lumen actually encloses."""
        return float(np.sqrt(max(self.area_um2, 0.0) / np.pi))

    @property
    def r_perimeter(self) -> float:
        """Perimeter-equivalent radius -- the convention the rest of the toolkit uses.

        For a collapsed vessel these two disagree strongly, and the disagreement is
        the point: a slit has a perimeter like the vessel it was and an area like
        nothing at all. See ``crosssection.py`` for where this convention comes from.
        """
        return float(self.perimeter_um / (2.0 * np.pi))

    @property
    def flatness(self) -> float:
        """1 for a disc, rising as the section flattens. ``r_perimeter / r_area``."""
        return float(self.r_perimeter / max(self.r_area, 1e-9))


@dataclass
class Completion:
    """The voxels a mask repair would add, and what shape they came from."""

    voxels_zyx: np.ndarray  # (M, 3) global segmentation indices
    core_voxels: int  # of which this many are the connectivity core
    sections: list  # the two measured ends
    radii_um: np.ndarray  # (N,) perimeter-equivalent radius along the route
    areas_um2: np.ndarray
    transported: bool = True
    reason: str = ""
    metrics: dict = field(default_factory=dict)

    def describe(self) -> str:
        if not len(self.voxels_zyx):
            return f"no mask completion: {self.reason or 'nothing to add'}"
        ends = ", ".join(f"r_per={s.r_perimeter:.0f}um flat={s.flatness:.1f}"
                         for s in self.sections if s.valid)
        return (f"{len(self.voxels_zyx):,} voxel(s) added "
                f"({self.core_voxels} as connectivity core)"
                + (f"; ends {ends}" if ends else ""))


# ------------------------------------------------------- rotation-minimizing frame


def rotation_minimizing_frame(points_um, initial_normal=None):
    """Tangents and a twist-free normal along a polyline (double reflection).

    Wang, Juttler, Zheng & Liu, *Computation of rotation minimizing frames*, ACM
    TOG 27(1), 2008. Two reflections carry the previous normal onto the next
    tangent's plane; the composition is a rotation, so the frame is transported
    with zero angular velocity about the tangent.

    Why not Frenet: its normal is defined by the curvature vector, so it is
    undefined on a straight run and flips through every inflection. A bridge is
    mostly straight with a couple of gentle inflections, which is the worst
    possible input for it -- the transported cross-section would snap through 180
    degrees partway across the gap.
    """
    points = np.asarray(points_um, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    if n < 2:
        t = np.array([[0.0, 0.0, 1.0]])
        return t, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])

    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-12)

    if initial_normal is None:
        seed = np.array([0.0, 0.0, 1.0])
        if abs(float(tangents[0] @ seed)) > 0.9:
            seed = np.array([1.0, 0.0, 0.0])
        normal = np.cross(tangents[0], seed)
    else:
        normal = np.asarray(initial_normal, dtype=np.float64)
        normal = normal - float(normal @ tangents[0]) * tangents[0]
    normal /= max(np.linalg.norm(normal), 1e-12)

    normals = np.zeros_like(points)
    normals[0] = normal
    for i in range(n - 1):
        v1 = points[i + 1] - points[i]
        c1 = float(v1 @ v1)
        if c1 < 1e-18:
            normals[i + 1] = normals[i]
            continue
        # First reflection: in the plane bisecting the two points.
        n_l = normals[i] - (2.0 / c1) * float(v1 @ normals[i]) * v1
        t_l = tangents[i] - (2.0 / c1) * float(v1 @ tangents[i]) * v1
        # Second: onto the next tangent.
        v2 = tangents[i + 1] - t_l
        c2 = float(v2 @ v2)
        if c2 < 1e-18:
            normals[i + 1] = n_l
        else:
            normals[i + 1] = n_l - (2.0 / c2) * float(v2 @ n_l) * v2
        normals[i + 1] /= max(np.linalg.norm(normals[i + 1]), 1e-12)

    binormals = np.cross(tangents, normals)
    return tangents, normals, binormals


# ------------------------------------------------------------- section extraction


def _plane_axes(tangent, seed=None):
    """Two orthonormal in-plane axes for a given tangent."""
    t = np.asarray(tangent, dtype=np.float64)
    t = t / max(np.linalg.norm(t), 1e-12)
    if seed is None:
        seed = np.array([0.0, 0.0, 1.0])
        if abs(float(t @ seed)) > 0.9:
            seed = np.array([1.0, 0.0, 0.0])
    u = np.asarray(seed, dtype=np.float64)
    u = u - float(u @ t) * t
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(t, np.array([1.0, 0.0, 0.0]))
    u /= max(np.linalg.norm(u), 1e-12)
    return u, np.cross(t, u)


def extract_section(index, frame, centre_um, tangent, component: int, radius_um: float,
                    *, seed_normal=None, half: int | None = None) -> Section:
    """The cross-section of one component, cut perpendicular to the vessel.

    Only voxels of `component` are kept. That restriction is what stops a cut that
    grazes a neighbouring vessel from importing its cross-section into the repair,
    which would then be transported across the gap as if it were this vessel's own
    shape.
    """
    spacing = np.asarray(frame.seg_spacing, dtype=np.float64)
    step = float(spacing.min())
    if half is None:
        half = int(max(np.ceil(SECTION_HALF_RADII * radius_um / step), SECTION_HALF_MIN))
    u, v = _plane_axes(tangent, seed_normal)

    offsets = (np.arange(-half, half + 1) * step)
    gu, gv = np.meshgrid(offsets, offsets, indexing="ij")
    points = (np.asarray(centre_um, dtype=np.float64)
              + gu[..., None] * u + gv[..., None] * v)
    ijk = np.asarray(frame.um_to_seg(points.reshape(-1, 3)), dtype=np.float64)
    zyx = np.round(ijk[:, ::-1]).astype(np.int64)
    labels = index.labels_at(zyx).reshape(gu.shape)
    blob = labels == int(component)

    if not blob.any():
        return Section(
            sdf=np.full(gu.shape, step, dtype=np.float32), u=u, v=v,
            tangent=np.asarray(tangent, dtype=np.float64),
            centre_um=np.asarray(centre_um, dtype=np.float64), step_um=step,
            area_um2=0.0, perimeter_um=0.0, valid=False,
            reason=f"component {component} does not intersect the cut plane",
        )
    blob = _central_blob(blob)
    sdf = signed_distance_2d(blob, step)
    area = float(blob.sum()) * step * step
    return Section(
        sdf=sdf.astype(np.float32), u=u, v=v,
        tangent=np.asarray(tangent, dtype=np.float64),
        centre_um=np.asarray(centre_um, dtype=np.float64), step_um=step,
        area_um2=area, perimeter_um=_perimeter(blob, step),
        valid=int(blob.sum()) >= MIN_SECTION_VOXELS,
    )


def _central_blob(blob: np.ndarray) -> np.ndarray:
    """Keep only the piece of the cut containing (or nearest) the centre.

    An oblique cut through a curving vessel can catch the same vessel twice. The
    piece the centreline is standing in is the one being transported.
    """
    from scipy import ndimage

    labels, n = ndimage.label(blob, structure=np.ones((3, 3), dtype=int))
    if n <= 1:
        return blob
    centre = tuple(s // 2 for s in blob.shape)
    here = int(labels[centre])
    if here:
        return labels == here
    coords = np.argwhere(blob)
    distance = np.linalg.norm(coords - np.asarray(centre), axis=1)
    return labels == int(labels[tuple(coords[int(np.argmin(distance))])])


def signed_distance_2d(blob: np.ndarray, step_um: float) -> np.ndarray:
    """Signed distance in um: negative inside the blob, positive outside."""
    from scipy import ndimage

    inside = np.asarray(blob, dtype=bool)
    if not inside.any():
        return np.full(inside.shape, step_um, dtype=np.float64)
    outer = ndimage.distance_transform_edt(~inside, sampling=step_um)
    inner = ndimage.distance_transform_edt(inside, sampling=step_um)
    return outer - inner


def _perimeter(blob: np.ndarray, step_um: float) -> float:
    """Boundary length of a 2D blob, in the toolkit's existing convention.

    Delegates to ``crosssection._perimeter_um``, which is ``cv2.arcLength`` over
    the external contour. Matching it is the whole point: ``r_perimeter`` written
    here has to be comparable with the one ``crosssection.measure`` and
    ``radius_perimeter`` report, and two estimators that disagree by 10-15% on the
    same blob would make a repaired vessel look like a calibre step.

    The fallback, when OpenCV is absent, counts exposed voxel faces. That is
    *exact* for an axis-aligned boundary where ``arcLength`` -- which traces voxel
    centres -- is short by one voxel width per side, so the two are not
    interchangeable and the difference is noted rather than smoothed over.
    """
    inside = np.asarray(blob, dtype=bool)
    if not inside.any():
        return 0.0
    try:
        from ....crosssection import _perimeter_um

        return float(_perimeter_um(inside, step_um))
    except Exception:  # noqa: BLE001 - OpenCV is optional; geometry is not
        faces = 4 * int(inside.sum()) - 2 * (
            int((inside[:, :-1] & inside[:, 1:]).sum())
            + int((inside[:-1] & inside[1:]).sum())
        )
        return float(max(faces, 0)) * step_um


# --------------------------------------------------------------------- transport


def transport(path_um, section_a: Section, section_b: Section, frame, *,
              min_radius_um: float = 0.0) -> Completion:
    """Sweep the interpolated cross-section along the route.

    Shape, orientation and scale are interpolated together: the two signed
    distance fields are blended linearly in arclength after each has been rotated
    into the transported frame, and the blend is scaled so the equivalent radius
    moves smoothly between the two measured ends rather than jumping at the join.
    """
    path = np.asarray(path_um, dtype=np.float64).reshape(-1, 3)
    if len(path) < 2:
        return Completion(np.empty((0, 3), np.int64), 0, [section_a, section_b],
                          np.array([]), np.array([]), transported=False,
                          reason="the route is too short to sweep")

    tangents, normals, binormals = rotation_minimizing_frame(
        path, initial_normal=section_a.u if section_a.valid else None
    )
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    s = arc / max(arc[-1], 1e-9)

    # Align each end's own (u, v) into the transported frame, so a blend of the
    # two fields is a blend of shapes rather than of two arbitrary orientations.
    rot_a = _alignment(section_a, normals[0], binormals[0], tangents[0])
    rot_b = _alignment(section_b, normals[-1], binormals[-1], tangents[-1])

    r_a = section_a.r_area if section_a.valid else min_radius_um
    r_b = section_b.r_area if section_b.valid else min_radius_um

    voxels: set[tuple[int, int, int]] = set()
    radii = np.zeros(len(path))
    areas = np.zeros(len(path))
    step_um = section_a.step_um if section_a.valid else section_b.step_um
    half = (section_a.sdf.shape[0] - 1) // 2

    offsets = np.arange(-half, half + 1) * step_um
    gu, gv = np.meshgrid(offsets, offsets, indexing="ij")

    for i in range(len(path)):
        w = float(s[i])
        field_a = _resample(section_a.sdf, rot_a) if section_a.valid else None
        field_b = _resample(section_b.sdf, rot_b) if section_b.valid else None
        blended = _blend(field_a, field_b, w, step_um)
        scale = ((1.0 - w) * r_a + w * r_b) / max(
            (1.0 - w) * (r_a or 1.0) + w * (r_b or 1.0), 1e-9
        )
        inside = blended <= 0.0
        if scale != 1.0:
            inside = blended <= (scale - 1.0) * step_um
        areas[i] = float(inside.sum()) * step_um * step_um
        radii[i] = _perimeter(inside, step_um) / (2.0 * np.pi)

        if not inside.any():
            continue
        local = (path[i]
                 + gu[inside][:, None] * normals[i]
                 + gv[inside][:, None] * binormals[i])
        ijk = np.asarray(frame.um_to_seg(local), dtype=np.float64)
        zyx = np.round(ijk[:, ::-1]).astype(np.int64)
        voxels.update(map(tuple, zyx.tolist()))

    swept = np.array(sorted(voxels), dtype=np.int64) if voxels else \
        np.empty((0, 3), dtype=np.int64)
    core = minimum_core(path, frame)
    added = _union(swept, core)
    return Completion(
        voxels_zyx=added, core_voxels=int(len(core)),
        sections=[section_a, section_b], radii_um=radii, areas_um2=areas,
        metrics={"swept_voxels": int(len(swept)),
                 "flatness_a": section_a.flatness if section_a.valid else 0.0,
                 "flatness_b": section_b.flatness if section_b.valid else 0.0},
    )


def _alignment(section: Section, normal, binormal, tangent) -> float:
    """The in-plane angle taking a section's own axes onto the transported frame."""
    if not section.valid:
        return 0.0
    return float(np.arctan2(float(section.u @ binormal), float(section.u @ normal)))


def _resample(sdf: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a signed distance field in its own plane by `angle` radians."""
    from scipy import ndimage

    if abs(angle) < 1e-6:
        return sdf
    # `reflect` rather than a constant: the field is a distance, and padding it
    # with a fixed number would place a fictitious surface at the window edge.
    return ndimage.rotate(sdf, np.degrees(angle), reshape=False, order=1,
                          mode="nearest")


def _blend(a, b, w: float, step_um: float) -> np.ndarray:
    if a is None and b is None:
        raise ValueError("neither end supplied a usable cross-section")
    if a is None:
        return np.asarray(b, dtype=np.float64)
    if b is None:
        return np.asarray(a, dtype=np.float64)
    if a.shape != b.shape:
        b = _fit(b, a.shape, step_um)
    return (1.0 - w) * np.asarray(a, dtype=np.float64) + w * np.asarray(b, np.float64)


def _fit(field: np.ndarray, shape, step_um: float) -> np.ndarray:
    """Pad or crop a signed distance field onto another window.

    Padding uses a large positive distance -- outside is outside -- rather than
    edge replication, which would extend whatever shape touched the border.
    """
    out = np.full(shape, float(max(shape)) * step_um, dtype=np.float64)
    n = [min(shape[k], field.shape[k]) for k in range(2)]
    src = [(field.shape[k] - n[k]) // 2 for k in range(2)]
    dst = [(shape[k] - n[k]) // 2 for k in range(2)]
    out[dst[0]:dst[0] + n[0], dst[1]:dst[1] + n[1]] = \
        field[src[0]:src[0] + n[0], src[1]:src[1] + n[1]]
    return out


def minimum_core(path_um, frame) -> np.ndarray:
    """The smallest 26-connected chain of voxels realising the route.

    The guarantee the whole repair rests on -- the two components must end up in
    one -- and deliberately the *minimum* that provides it. Where the transported
    section already covers the route this adds nothing; where the section
    collapses to nothing (a slit thinner than a voxel, which does happen) this is
    the single line of voxels that keeps the lumen continuous, and it is honest
    about being the smallest possible claim.
    """
    path = np.asarray(path_um, dtype=np.float64).reshape(-1, 3)
    if len(path) < 1:
        return np.empty((0, 3), dtype=np.int64)
    ijk = np.asarray(frame.um_to_seg(path), dtype=np.float64)
    zyx = np.round(ijk[:, ::-1]).astype(np.int64)

    out = [zyx[0]]
    for target in zyx[1:]:
        while True:
            delta = target - out[-1]
            if not np.any(delta):
                break
            out.append(out[-1] + np.clip(delta, -1, 1))
    seen: dict[tuple, None] = {}
    for voxel in out:
        seen.setdefault(tuple(int(v) for v in voxel), None)
    return np.array(list(seen), dtype=np.int64)


def _union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a):
        return b
    if not len(b):
        return a
    return np.unique(np.vstack([a, b]), axis=0)


def measure_ends(index, frame, source, target, *, seed_normal=None
                 ) -> tuple[Section, Section]:
    """Cut both intact ends, sharing a seed normal so their frames start aligned."""
    a = extract_section(index, frame, source.point_um, source.tangent,
                        source.component, source.radius_um, seed_normal=seed_normal)
    b = extract_section(index, frame, target.point_um, -np.asarray(target.tangent),
                        target.component, target.radius_um,
                        seed_normal=a.u if a.valid else seed_normal)
    return a, b


def graph_radii(completion: Completion, source_radius: float, target_radius: float,
                n: int) -> np.ndarray:
    """Perimeter-equivalent radii to write on the *graph*, along an n-point route.

    Separate from the mask on purpose. The mask says what the collapsed specimen
    looks like; this says what calibre the vessel has, in the perimeter-equivalent
    convention ``crosssection.py`` and ``radius_perimeter.py`` already use and that
    the SDF surface pipeline consumes. A route whose transported sections measured
    well uses those measurements; one whose sections were unusable falls back to
    interpolating the two endpoint radii, and the fallback is visible in the audit
    rather than silently indistinguishable.
    """
    measured = np.asarray(completion.radii_um, dtype=np.float64)
    measured = measured[np.isfinite(measured) & (measured > 0)]
    if len(measured) >= 2:
        return np.interp(np.linspace(0.0, 1.0, n),
                         np.linspace(0.0, 1.0, len(measured)), measured)
    return np.linspace(float(source_radius), float(target_radius), n)
