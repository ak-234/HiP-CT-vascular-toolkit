"""Undo/redo, and the record of *where* an edit landed.

Every edit reports a :class:`Patch`: which segments it touched and the world
box it touched them in. That box is the whole point -- it is what
:mod:`.sdfpatch` rebuilds, and rebuilding a 10 mm cube instead of the whole tree
is the difference between a sub-second preview and a ten-minute one.

Commands store an explicit inverse rather than a snapshot of the graph. A
snapshot would be simpler, but the graph holds millions of points and a copy per
keystroke is not affordable; the touched state is a handful of tuples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .graphmodel import EditableGraph


@dataclass(frozen=True)
class Patch:
    """Segments touched by an edit, and the world box (um) they were touched in.

    ``aabb`` is ``(2, 3)`` -- ``[[xmin, ymin, zmin], [xmax, ymax, zmax]]`` -- or
    ``None`` for an edit with no spatial extent (a pure topology relabel). It is
    the *pre- and post-edit* extent combined: moving a point has to invalidate
    where it was as well as where it went.
    """

    seg_ids: frozenset[int]
    aabb: np.ndarray | None

    @classmethod
    def empty(cls) -> "Patch":
        return cls(frozenset(), None)

    @classmethod
    def of(cls, seg_ids: Iterable[int], coords: Sequence | np.ndarray | None) -> "Patch":
        """Build from segment ids and any set of world points to be enclosed."""
        box = None
        if coords is not None:
            pts = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
            if len(pts):
                box = np.array([pts.min(axis=0), pts.max(axis=0)])
        return cls(frozenset(int(s) for s in seg_ids), box)

    def merged(self, other: "Patch") -> "Patch":
        """Union of two patches -- used to coalesce a burst of edits into one rebuild."""
        if self.aabb is None:
            box = other.aabb
        elif other.aabb is None:
            box = self.aabb
        else:
            box = np.array(
                [
                    np.minimum(self.aabb[0], other.aabb[0]),
                    np.maximum(self.aabb[1], other.aabb[1]),
                ]
            )
        return Patch(self.seg_ids | other.seg_ids, box)

    def padded(self, pad: float) -> np.ndarray | None:
        """The box grown by `pad` in every direction (um)."""
        if self.aabb is None:
            return None
        return np.array([self.aabb[0] - pad, self.aabb[1] + pad])

    @property
    def is_empty(self) -> bool:
        return not self.seg_ids and self.aabb is None


class Command(ABC):
    """One reversible edit. ``do`` and ``undo`` both report where they landed."""

    label: str = "edit"

    @abstractmethod
    def do(self, graph: "EditableGraph") -> Patch: ...

    @abstractmethod
    def undo(self, graph: "EditableGraph") -> Patch: ...

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.label!r}>"


class Composite(Command):
    """Several commands applied as one undo step.

    Reconnection scripts emit dozens of edits that only make sense together;
    undoing half a bridge is never what anyone wants.
    """

    def __init__(self, commands: Sequence[Command], label: str = "batch"):
        self.commands = list(commands)
        self.label = label

    def do(self, graph: "EditableGraph") -> Patch:
        patch = Patch.empty()
        for cmd in self.commands:
            patch = patch.merged(cmd.do(graph))
        return patch

    def undo(self, graph: "EditableGraph") -> Patch:
        patch = Patch.empty()
        for cmd in reversed(self.commands):
            patch = patch.merged(cmd.undo(graph))
        return patch


class History:
    """A bounded undo/redo stack over one :class:`~.graphmodel.EditableGraph`."""

    def __init__(self, graph: "EditableGraph", limit: int = 200):
        self.graph = graph
        self.limit = limit
        self._done: list[Command] = []
        self._undone: list[Command] = []

    def run(self, command: Command) -> Patch:
        """Execute `command`, push it, and discard any redo branch."""
        patch = command.do(self.graph)
        self.push_done(command)
        return patch

    def push_done(self, command: Command) -> None:
        """Record a command that has *already* been applied.

        Used by :meth:`~.graphmodel.EditableGraph.batch`, where the individual
        edits must run as they are issued -- each one needs to see the previous
        one's state -- but should collapse into a single undo step.
        """
        self._done.append(command)
        self._undone.clear()
        if len(self._done) > self.limit:
            # Dropping the oldest command makes it permanent, which is the
            # normal meaning of an undo limit.
            del self._done[0]

    def undo(self) -> Patch | None:
        if not self._done:
            return None
        command = self._done.pop()
        patch = command.undo(self.graph)
        self._undone.append(command)
        return patch

    def redo(self) -> Patch | None:
        if not self._undone:
            return None
        command = self._undone.pop()
        patch = command.do(self.graph)
        self._done.append(command)
        return patch

    @property
    def can_undo(self) -> bool:
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        return bool(self._undone)

    @property
    def undo_label(self) -> str | None:
        return self._done[-1].label if self._done else None

    @property
    def redo_label(self) -> str | None:
        return self._undone[-1].label if self._undone else None

    def labels(self) -> list[str]:
        return [c.label for c in self._done]

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()
