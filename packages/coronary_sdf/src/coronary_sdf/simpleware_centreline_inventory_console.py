"""Read-only inventory of persisted Simpleware centreline network topology."""

import os
import sys
import traceback
from collections import Counter
from pathlib import Path

from simpleware.scripting import App


LOG_PATH = Path(
    os.environ.get("CORONARY_SDF_LOG_DIR") or (Path.cwd() / "logs")
) / "simpleware_centreline_inventory.log"


def _log(message):
    line = "[CENTRELINE] {}".format(message)
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()


def main():
    app = App.GetInstance()
    input_value = str(app.GetInputValue() or "").strip().strip('"')
    input_value = input_value or os.environ.get(
        "SIMPLEWARE_INVENTORY_PROJECT", ""
    )
    project_path = Path(input_value).resolve()
    if not project_path.is_file():
        raise RuntimeError("Project not found: {}".format(project_path))

    _log("Opening project read-only: {}".format(project_path))
    document = app.OpenDocument(str(project_path))
    networks = list(document.GetCentrelines().GetNetworks(False))
    _log("Persisted network count: {}".format(len(networks)))
    for network in networks:
        nodes = list(network.GetNodes())
        splines = list(network.GetSplines())
        degree_counts = Counter(len(list(node.GetSplines())) for node in nodes)
        terminal_nodes = [node for node in nodes if len(list(node.GetSplines())) == 1]
        raw_counts = []
        terminal_lengths = []
        for node in terminal_nodes:
            spline = list(node.GetSplines())[0]
            raw_counts.append(len(list(spline.GetRawDataPoints(False))))
            terminal_lengths.append(float(spline.GetLength(False)))
        raw_summary = (
            "{}..{}".format(min(raw_counts), max(raw_counts))
            if raw_counts else "none"
        )
        length_summary = (
            "{:.3f}..{:.3f} mm".format(
                min(terminal_lengths), max(terminal_lengths)
            )
            if terminal_lengths else "none"
        )
        _log(
            "network={!r}, visible={}, nodes={}, splines={}, degrees={}, "
            "degree-one terminals={}, terminal raw-point range={}, terminal "
            "spline-length range={}".format(
                network.GetName(),
                bool(network.GetVisible()),
                len(nodes),
                len(splines),
                dict(sorted(degree_counts.items())),
                len(terminal_nodes),
                raw_summary,
                length_summary,
            )
        )


if __name__ == "__main__":
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("", encoding="utf-8")
        main()
    except Exception:
        failure = traceback.format_exc()
        print(failure, flush=True)
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(failure)
            stream.flush()
        sys.stderr.flush()
        raise
