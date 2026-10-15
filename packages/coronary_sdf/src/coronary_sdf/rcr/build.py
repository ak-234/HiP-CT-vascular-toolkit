"""Build only the exported source copies with the selected CFX installation."""
from pathlib import Path
import subprocess

from .prepare import digest, write_json


def build_libraries(directory, cfx_pre, *, double=False, timeout=120):
    directory = Path(directory).resolve()
    executable = Path(cfx_pre).with_name("cfx5mkext.exe" if Path(cfx_pre).suffix == ".exe" else "cfx5mkext")
    if not executable.is_file():
        raise ValueError("Selected CFX installation has no cfx5mkext")
    hashes = {str(f.relative_to(directory)): digest(f) for f in directory.rglob("*.F")}
    binaries = {}
    commands = []
    for folder, name in (("jbpandq", "jbpandq"), ("jbqpast", "jbqpast"), ("windk", "windkessel")):
        target = directory/folder
        command = [str(executable), "-name", name, "-dest", str(target)]
        if double:
            command.append("-double")
        command.append(str(target/f"{name}.F"))
        commands.append(command)
        try:
            with (target/"build.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=target, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"Compilation timed out; inspect {target/'build.log'}") from exc
        outputs = [p for p in target.rglob("*") if p.suffix.lower() in (".dll", ".so")]
        if result.returncode or not outputs:
            raise ValueError(f"Compilation failed; inspect {target/'build.log'}")
        binaries.update({str(p.relative_to(directory)): digest(p) for p in outputs})
    manifest = dict(source_sha256=hashes, binary_sha256=binaries, commands=commands,
                    double_precision=double, cfx_pre=str(cfx_pre), solver_validated=False)
    write_json(directory/"build_manifest.json", manifest)
    return manifest
