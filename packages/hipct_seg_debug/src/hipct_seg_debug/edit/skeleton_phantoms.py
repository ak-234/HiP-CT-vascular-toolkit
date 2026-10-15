"""Vessel phantoms with a known centreline, radius and collapse ratio.

A skeleton of real HiP-CT data cannot be scored for position -- there is no ground
truth, which is why the super metric scores against the mask instead. These can be:
every phantom is swept from analytic centrelines, so its centreline, topology,
inflated radius ``R0`` and collapse ratio ``lambda = b/a`` are known at every sample.

Collapse follows the inextensible-perimeter tube law the radius pipeline assumes: a
section of inflated radius ``R0`` flattens to the ellipse with the *same perimeter*
``2 pi R0``. Ramanujan's perimeter approximation gives the semi-axes in closed form,
``a = 2 R0 / h(lambda)``, ``b = lambda a`` with
``h(lambda) = 3(1 + lambda) - sqrt((3 + lambda)(1 + 3 lambda))`` -- at ``lambda = 1``
that is the circle of radius ``R0``. Its error is under 0.01% at every ratio here.

Coordinates are ``(z, y, x)`` voxels throughout, matching the mask. The ellipse's
major axis is a unit vector ``u`` perpendicular to the tangent at each sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PHANTOMS = ("round", "collapse-bands", "curved-collapsed", "y-bifurcation", "loop",
            "protrusions")
COLLAPSE_BANDS = (1.0, 0.5, 0.3, 0.2, 0.1)
SPACING_UM = 10.0
SAMPLE_STEP = 0.25  # voxels between centreline samples used for rasterising


def ramanujan_h(lam):
    lam = np.asarray(lam, dtype=np.float64)
    return 3 * (1 + lam) - np.sqrt((3 + lam) * (1 + 3 * lam))


def semi_axes(r0, lam):
    """Perimeter-conserving ellipse ``(a, b)`` for inflated radius `r0`, ratio `lam`."""
    a = 2 * np.asarray(r0, dtype=np.float64) / ramanujan_h(lam)
    return a, np.asarray(lam) * a


def ellipse_perimeter(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


def analytic_radius_table(lams=COLLAPSE_BANDS) -> list[dict]:
    """What each radius estimator returns, relative to ``R0``, on a collapsed lumen.

    ``inscribed`` is the inscribed-ball radius ``b`` (every distance-transform radius),
    ``area`` is ``sqrt(ab)``, and ``perimeter`` is exact by construction.
    """
    rows = []
    for lam in lams:
        a, b = semi_axes(1.0, lam)
        rows.append(dict(lam=float(lam), inscribed=float(b), area=float(np.sqrt(a * b)),
                         perimeter=1.0, resistance_inscribed=float(b ** -4)))
    return rows


@dataclass
class Branch:
    """One analytic centreline: samples, inflated radius, collapse ratio, major axis."""

    points: np.ndarray  # (n, 3) zyx voxels
    r0: np.ndarray  # (n,) voxels
    lam: np.ndarray  # (n,)
    major: np.ndarray  # (n, 3) unit, perpendicular to the tangent
    node_a: int
    node_b: int


@dataclass
class Phantom:
    name: str
    mask: np.ndarray  # (nz, ny, nx) uint8
    spacing_um: float
    nodes: np.ndarray  # (K, 3) zyx voxels
    branches: list[Branch]
    root_zyx: tuple[int, int, int]
    components: int = 1
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def edges(self) -> list[tuple[int, int]]:
        return [(b.node_a, b.node_b) for b in self.branches]

    def degrees(self) -> np.ndarray:
        deg = np.zeros(len(self.nodes), dtype=int)
        for a, b in self.edges:
            deg[a] += 1
            deg[b] += 1
        return deg

    @property
    def terminals(self) -> np.ndarray:
        return self.nodes[self.degrees() == 1]

    @property
    def junctions(self) -> np.ndarray:
        return self.nodes[self.degrees() >= 3]

    @property
    def beta1(self) -> int:
        return len(self.branches) - len(self.nodes) + self.components

    def samples(self, step: float = 0.5):
        """Dense GT samples: ``(points zyx, r0, lam, branch index)``."""
        pts, r0, lam, bid = [], [], [], []
        for i, br in enumerate(self.branches):
            s = _arclength(br.points)
            n = max(2, int(np.ceil(s[-1] / step)) + 1)
            t = np.linspace(0, s[-1], n)
            pts.append(np.column_stack([np.interp(t, s, br.points[:, k]) for k in range(3)]))
            r0.append(np.interp(t, s, br.r0))
            lam.append(np.interp(t, s, br.lam))
            bid.append(np.full(n, i))
        return (np.vstack(pts), np.concatenate(r0), np.concatenate(lam),
                np.concatenate(bid))


def _arclength(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def _resample(points, step=SAMPLE_STEP):
    s = _arclength(points)
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, points[:, k]) for k in range(3)]), t / s[-1]


def _tangents(points):
    t = np.gradient(points, axis=0)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def _perpendicular(t, hint):
    """`hint` with its tangential part removed, normalised (falls back if parallel)."""
    hint = np.broadcast_to(np.asarray(hint, dtype=np.float64), t.shape)
    u = hint - np.sum(hint * t, axis=1, keepdims=True) * t
    bad = np.linalg.norm(u, axis=1) < 1e-6
    if bad.any():
        alt = np.cross(t[bad], [1.0, 0.0, 0.0])
        alt[np.linalg.norm(alt, axis=1) < 1e-6] = np.cross(
            t[bad][np.linalg.norm(alt, axis=1) < 1e-6], [0.0, 1.0, 0.0])
        u[bad] = alt
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def make_branch(points, r0, lam, node_a, node_b, *, major_hint=(1.0, 0.0, 0.0),
                twist=0.0) -> Branch:
    """Resample a polyline and attach radius, ratio and major axis.

    `r0` and `lam` are scalars or callables of normalised arclength in [0, 1]. The
    major axis is `major_hint` projected off the tangent, then rotated about the
    tangent by ``twist * s`` radians.
    """
    pts, s = _resample(np.asarray(points, dtype=np.float64))
    r = np.asarray(r0(s) if callable(r0) else np.full(len(s), r0), dtype=np.float64)
    la = np.asarray(lam(s) if callable(lam) else np.full(len(s), lam), dtype=np.float64)
    t = _tangents(pts)
    u = _perpendicular(t, major_hint)
    if twist:
        v = np.cross(t, u)
        phi = (twist * s)[:, None]
        u = np.cos(phi) * u + np.sin(phi) * v
    return Branch(pts, r, la, u, node_a, node_b)


def rasterise(shape, branches, *, supersample: int = 2, caps=None) -> np.ndarray:
    """Sweep each branch's ellipse along its centreline into a ``uint8`` mask.

    Each voxel is split into ``supersample**3`` sub-voxels; a sub-voxel is inside a
    branch when, relative to its nearest centreline sample, it lies within half a
    sample step axially (or anywhere ahead of an interior sample) and inside that
    sample's ellipse. A voxel is lumen when at least half its sub-voxels are.
    `caps` is ``{(branch index, 0 or -1): False}`` to leave an end open, which is how
    a daughter is joined to its parent without a flat face inside the junction.

    Closed ends are **rounded**: a half-ellipsoid with semi-axes ``(a, b, a)`` beyond
    the end sample -- a hemisphere on a round tube. A flat cap is not what a vessel
    looks like, and its medial set forks into the cap's corners, so on a slit every
    method grew a "false branch" that was really the phantom's flat face.
    """
    from scipy.spatial import cKDTree

    caps = caps or {}
    out = np.zeros(shape, dtype=np.float32)
    ss = int(supersample)
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    sub = np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), -1).reshape(-1, 3)
    for bi, br in enumerate(branches):
        a, b = semi_axes(br.r0, br.lam)
        reach = float(np.max(a)) + 1.5
        lo = np.maximum(np.floor(br.points.min(0) - reach).astype(int), 0)
        hi = np.minimum(np.ceil(br.points.max(0) + reach).astype(int) + 1, shape)
        if np.any(hi <= lo):
            continue
        grid = np.stack(np.meshgrid(*[np.arange(lo[k], hi[k]) for k in range(3)],
                                    indexing="ij"), -1).reshape(-1, 3).astype(np.float64)
        tree = cKDTree(br.points)
        t = _tangents(br.points)
        v_axis = np.cross(t, br.major)
        hits = np.zeros(len(grid), dtype=np.int32)
        for o in sub:
            q = grid + o
            dist, k = tree.query(q, distance_upper_bound=reach)
            ok = np.isfinite(dist)
            kk = np.where(ok, k, 0)
            d = q - br.points[kk]
            axial = np.sum(d * t[kk], axis=1)
            du = np.sum(d * br.major[kk], axis=1)
            dv = np.sum(d * v_axis[kk], axis=1)
            section = (du / a[kk]) ** 2 + (dv / b[kk]) ** 2
            inside = ok & (section <= 1.0)
            dome = section + (axial / a[kk]) ** 2 <= 1.0
            first, last = kk == 0, kk == len(br.points) - 1
            beyond0 = first & (axial < -0.5 * SAMPLE_STEP)
            beyond1 = last & (axial > 0.5 * SAMPLE_STEP)
            if caps.get((bi, 0), True):
                inside &= ~beyond0 | dome
            else:
                inside &= ~(first & (axial < -2 * float(np.max(a))))
            if caps.get((bi, -1), True):
                inside &= ~beyond1 | dome
            else:
                inside &= ~(last & (axial > 2 * float(np.max(a))))
            hits += inside
        frac = (hits / len(sub)).reshape(hi - lo)
        view = out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.maximum(view, frac, out=view)
    return (out >= 0.5).astype(np.uint8)


def _inside_root(mask, node, direction):
    """The first lumen voxel from a terminal node stepping along `direction`."""
    p = np.asarray(node, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    for step in np.arange(0.0, 10.0, 0.5):
        q = tuple(np.rint(p + step * d).astype(int))
        if all(0 <= q[k] < mask.shape[k] for k in range(3)) and mask[q]:
            return tuple(int(v) for v in q)
    raise ValueError("phantom root is not inside its own mask")


def _straight(x0, x1, cy, cz):
    return np.array([[cz, cy, x0], [cz, cy, x1]], dtype=np.float64)


def _smooth_bands(bands, ramp=0.04):
    """Piecewise-constant lambda over equal arclength bands, cosine-blended at edges."""
    bands = np.asarray(bands, dtype=np.float64)
    edges = np.linspace(0, 1, len(bands) + 1)

    def lam(s):
        s = np.asarray(s)
        out = np.full(s.shape, bands[0])
        for i in range(1, len(bands)):
            w = np.clip((s - (edges[i] - ramp)) / (2 * ramp), 0, 1)
            w = 0.5 - 0.5 * np.cos(np.pi * w)
            out = out + w * (bands[i] - bands[i - 1])
        return out
    return lam, edges


def build(name: str, *, spacing_um: float = SPACING_UM, seed: int = 0) -> Phantom:
    """One named phantom. See :data:`PHANTOMS`."""
    if name == "round":
        shape = (30, 30, 120)
        br = [make_branch(_straight(10, 110, 15, 15), 5.0, 1.0, 0, 1)]
        m = rasterise(shape, br)
        nodes = br[0].points[[0, -1]]
        return Phantom(name, m, spacing_um, nodes, br, _inside_root(m, nodes[0], (0, 0, 1)),
                       notes="straight round tube, R0 = 5 voxels")

    if name == "collapse-bands":
        lam, edges = _smooth_bands(COLLAPSE_BANDS)
        # R0 = 8 keeps the lambda = 0.1 slit 2.5 voxels thick; at 6 it is under two and
        # the voxel area overshoots pi*a*b by half, which tests rasterisation, not methods.
        shape = (28, 56, 256)
        br = [make_branch(_straight(16, 236, 28, 14), 8.0, lam, 0, 1, major_hint=(0, 1, 0))]
        m = rasterise(shape, br)
        nodes = br[0].points[[0, -1]]
        return Phantom(name, m, spacing_um, nodes, br, _inside_root(m, nodes[0], (0, 0, 1)),
                       notes="straight, R0 = 8 voxels, lambda bands "
                             + ", ".join(f"{v:g}" for v in COLLAPSE_BANDS),
                       extra=dict(band_edges=edges.tolist()))

    if name == "curved-collapsed":
        shape = (40, 64, 190)
        x = np.linspace(12, 178, 400)
        y = 32 + 12 * np.sin((x - 12) / 166 * 2 * np.pi)
        pts = np.column_stack([np.full_like(x, 20.0), y, x])
        br = [make_branch(pts, 6.0, 0.3, 0, 1, major_hint=(1, 0, 0), twist=np.pi)]
        m = rasterise(shape, br)
        nodes = br[0].points[[0, -1]]
        return Phantom(name, m, spacing_um, nodes, br, _inside_root(m, nodes[0], (0, 0, 1)),
                       notes="sinusoid, R0 = 6 voxels, lambda 0.3, major axis half-twist")

    if name == "y-bifurcation":
        shape = (40, 110, 150)
        j = np.array([20.0, 55.0, 75.0])
        parent = np.array([[20.0, 55.0, 10.0], j])
        ang = np.radians(35)
        d1 = j + 65 * np.array([0, np.sin(ang), np.cos(ang)])
        d2 = j + 65 * np.array([0, -np.sin(ang), np.cos(ang)])
        rd = 6.0 / 2 ** (1 / 3)
        br = [make_branch(parent, 6.0, 1.0, 0, 1),
              make_branch(np.array([j, d1]), rd, 1.0, 1, 2),
              make_branch(np.array([j, d2]), rd, 0.3, 1, 3, major_hint=(1, 0, 0))]
        m = rasterise(shape, br, caps={(0, -1): False, (1, 0): False, (2, 0): False})
        nodes = np.array([parent[0], j, d1, d2])
        return Phantom(name, m, spacing_um, nodes, br, _inside_root(m, nodes[0], (0, 0, 1)),
                       notes="round parent R0 = 6, Murray daughters; daughter 2 lambda 0.3")

    if name == "loop":
        shape = (24, 80, 170)
        cz, cy, r = 12.0, 40.0, 24.0
        j1, j2 = np.array([cz, cy, 50.0]), np.array([cz, cy, 50.0 + 2 * r])
        th = np.linspace(np.pi, 0, 200)
        upper = np.column_stack([np.full_like(th, cz), cy + r * np.sin(th),
                                 50.0 + r + r * np.cos(th)])
        lower = upper.copy()
        lower[:, 1] = cy - r * np.sin(th)
        br = [make_branch(np.array([[cz, cy, 8.0], j1]), 4.0, 1.0, 0, 1),
              make_branch(upper, 4.0, 1.0, 1, 2),
              make_branch(lower, 4.0, 1.0, 1, 2),
              make_branch(np.array([j2, [cz, cy, 160.0]]), 4.0, 1.0, 2, 3)]
        caps = {(0, -1): False, (1, 0): False, (1, -1): False, (2, 0): False,
                (2, -1): False, (3, 0): False}
        m = rasterise(shape, br, caps=caps)
        nodes = np.array([[cz, cy, 8.0], j1, j2, [cz, cy, 160.0]])
        return Phantom(name, m, spacing_um, nodes, br, _inside_root(m, nodes[0], (0, 0, 1)),
                       notes="anastomosis ring, R0 = 4 voxels, beta1 = 1")

    if name == "protrusions":
        shape = (30, 30, 120)
        br = [make_branch(_straight(10, 110, 15, 15), 5.0, 1.0, 0, 1)]
        m = rasterise(shape, br).astype(bool)
        rng = np.random.default_rng(seed)
        zz, yy, xx = np.indices(shape)
        for k in range(20):
            x = rng.uniform(25, 95)
            th = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(1.5, 2.5)
            c = np.array([15 + 5 * np.sin(th), 15 + 5 * np.cos(th), x])
            ball = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= rad ** 2
            if k < 12:
                m |= ball
            else:
                m &= ~ball
        from scipy import ndimage
        lab, _ = ndimage.label(m, structure=np.ones((3, 3, 3)))
        m = lab == lab[15, 15, 60]
        nodes = br[0].points[[0, -1]]
        return Phantom(name, m.astype(np.uint8), spacing_um, nodes, br,
                       _inside_root(m, nodes[0], (0, 0, 1)),
                       notes="round tube with 12 bumps and 8 dents of radius 1.5-2.5 voxels")

    raise ValueError(f"unknown phantom {name!r}; choose from {PHANTOMS}")
