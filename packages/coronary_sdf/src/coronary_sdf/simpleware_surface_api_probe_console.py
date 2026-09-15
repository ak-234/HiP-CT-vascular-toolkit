"""Read-only ConsoleSimpleware probe for active surface triangle access."""

import sys
from pathlib import Path

from simpleware.scripting import App


def main():
    app = App.GetInstance()
    project = Path(str(app.GetInputValue()).strip().strip('"')).resolve()
    document = app.OpenDocument(str(project))
    surfaces = list(document.GetSurfaces())
    print("SURFACES", len(surfaces), flush=True)
    for surface in surfaces:
        print("SURFACE", surface.GetName(), type(surface), flush=True)
        names = sorted(set(dir(surface)))
        print("METHODS", " ".join(names), flush=True)
        interesting = [
            name for name in names
            if any(token in name.lower() for token in (
                "point", "element", "triangle", "cell", "face", "export",
                "write", "save", "mesh", "connect", "polygon"
            ))
        ]
        print("INTERESTING", " ".join(interesting), flush=True)
        for name in interesting:
            try:
                value = getattr(surface, name)
                print("ATTR", name, repr(value), flush=True)
            except Exception as exc:
                print("ATTR_ERROR", name, type(exc).__name__, str(exc), flush=True)


if __name__ == "__main__":
    main()
