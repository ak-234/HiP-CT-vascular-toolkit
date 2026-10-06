"""Refresh the relocated local coronary study using full-segment point WSS."""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "coronary_sdf" / "src"))
from coronary_sdf.plane_sensitivity import run_plane_sensitivity


def main():
    config_path = Path(sys.argv[1]).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
    local = config_path.parent
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = value.replace("F:\\coronary_sdf\\", str(local) + "\\")
    output = Path(cfg["paths"]["output_dir"])
    backup = output / "analysis_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup.mkdir(parents=True)
    for relative in ("plane_sensitivity/reports", "plane_sensitivity/all_vessels/reports",
                     "plane_sensitivity/all_vessels_adaptive/reports"):
        source = output / relative
        if source.is_dir():
            shutil.copytree(source, backup / relative)
    for name in ("plane_sensitivity.py", "plane_sensitivity_README.md"):
        source = local / name
        if source.exists():
            shutil.copy2(source, backup / name)
        updated = REPO / "packages" / "coronary_sdf"
        if name.endswith(".py"):
            updated = updated / "src" / "coronary_sdf"
        shutil.copy2(updated / name, source)
    (backup / "effective_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("Previous reports backed up to", backup, flush=True)
    print(run_plane_sensitivity(cfg, resume=True), flush=True)


if __name__ == "__main__":
    main()
