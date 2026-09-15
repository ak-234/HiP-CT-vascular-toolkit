"""Per-component completion manifest for a reconstruction run.

A partial output directory used to be indistinguishable from a legitimate
single-component result: component 0 is written without a ``_g`` suffix, so a
run that aborted on component 1 leaves exactly the files a one-component graph
would leave. This module records what was *requested* alongside what was
*completed*, and marks an incomplete directory explicitly.

Kept separate from :mod:`coronary_sdf.pipeline` so the benchmark and the tests
can import the schema without importing the heavy reconstruction stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Telemetry keys attached to a surface's ``field_data`` by the four backends.
# Single source of truth: the benchmark sums these across components while the
# manifest keeps them attributed per component.
TELEMETRY_KEYS = (
    "field_evaluations",
    "hierarchy_seconds",
    "field_seconds",
    "extraction_seconds",
    "total_seconds",
)

# Resolution descriptors. Dense and adaptive backends parameterise resolution on
# different axes, so both are recorded and promoted out of the summed telemetry
# into their own manifest fields; without them the two families cannot be placed
# on one error-versus-resolution comparison.
RESOLUTION_KEYS = (
    "voxel_size_mm",
    "cells_across_diameter",
    "length_scale_mm",
)

MANIFEST_NAME = "component_manifest.json"
INCOMPLETE_MARKER = "INCOMPLETE"


def component_suffix(graph_id: int) -> str:
    """Output filename suffix for ``graph_id``.

    Component 0 is deliberately unsuffixed; the README, existing output
    directories and the downstream tools all depend on that.
    """

    return f"_g{graph_id}" if graph_id > 0 else ""


def surface_telemetry(surface) -> dict[str, float]:
    """Extract the per-component telemetry a backend attached to ``surface``."""

    if surface is None:
        return {}
    data = getattr(surface, "field_data", None)
    if data is None:
        return {}
    result: dict[str, float] = {}
    for key in TELEMETRY_KEYS + RESOLUTION_KEYS:
        if key in data:
            values = data[key]
            if len(values):
                result[key] = float(values[0])
    return result


@dataclass(frozen=True)
class ComponentReport:
    """Outcome of reconstructing one connected component."""

    graph_id: int
    status: str  # "ok" | "failed" | "empty"
    n_nodes: int = 0
    n_points: int = 0
    n_segments: int = 0
    segment_ids: tuple[int, ...] = ()
    mesh_points: int = 0
    mesh_faces: int = 0
    field_method: str | None = None
    mesh_method: str | None = None
    primitive_method: str | None = None
    voxel_size_mm: float | None = None
    cells_across_diameter: float | None = None
    length_scale_mm: float | None = None
    telemetry: dict[str, float] = field(default_factory=dict)
    output_paths: tuple[str, ...] = ()
    validation: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    error_type: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineReport:
    """Manifest of every component a run requested, completed and failed."""

    schema_version: int
    graph_source: str
    output_dir: str
    config_digest: str
    created_utc: str
    requested_components: tuple[int, ...]
    completed_components: tuple[int, ...]
    failed_components: tuple[int, ...]
    components: tuple[ComponentReport, ...]

    @property
    def incomplete(self) -> bool:
        return set(self.completed_components) != set(self.requested_components)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["incomplete"] = self.incomplete
        return result

    def write(self, output_dir: str | Path) -> Path:
        """Write the manifest, plus a plain-text marker when incomplete."""

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / MANIFEST_NAME
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        marker = out / INCOMPLETE_MARKER
        if self.incomplete:
            missing = sorted(
                set(self.requested_components) - set(self.completed_components)
            )
            lines = [
                "This reconstruction did not emit every requested graph component.",
                f"requested: {list(self.requested_components)}",
                f"completed: {list(self.completed_components)}",
                f"missing:   {missing}",
                "",
            ]
            for report in self.components:
                if report.status != "ok":
                    lines.append(
                        f"  graph {report.graph_id}: {report.status} "
                        f"{report.error_type or ''} {report.error or ''}".rstrip()
                    )
            marker.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif marker.exists():
            # A previous incomplete run must not leave a stale marker behind.
            marker.unlink()
        return path


class PipelineReportBuilder:
    """Accumulates :class:`ComponentReport` entries during a run."""

    def __init__(
        self,
        *,
        requested: list[int] | tuple[int, ...],
        graph_source: str,
        output_dir: str | Path,
        config_digest: str = "",
    ) -> None:
        self._requested = tuple(int(value) for value in requested)
        self._graph_source = str(graph_source)
        self._output_dir = str(output_dir)
        self._config_digest = config_digest
        self._components: list[ComponentReport] = []

    def record(self, report: ComponentReport) -> None:
        self._components.append(report)

    def finish(self) -> PipelineReport:
        completed = tuple(
            report.graph_id for report in self._components if report.status == "ok"
        )
        failed = tuple(
            report.graph_id for report in self._components if report.status == "failed"
        )
        return PipelineReport(
            schema_version=SCHEMA_VERSION,
            graph_source=self._graph_source,
            output_dir=self._output_dir,
            config_digest=self._config_digest,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            requested_components=self._requested,
            completed_components=completed,
            failed_components=failed,
            components=tuple(self._components),
        )


def config_digest(cfg) -> str:
    """Stable digest of a configuration, for provenance in the manifest."""

    try:
        payload = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    except Exception:
        return ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "INCOMPLETE_MARKER",
    "MANIFEST_NAME",
    "RESOLUTION_KEYS",
    "SCHEMA_VERSION",
    "TELEMETRY_KEYS",
    "ComponentReport",
    "PipelineReport",
    "PipelineReportBuilder",
    "component_suffix",
    "config_digest",
    "surface_telemetry",
]
