"""Gather the repository's Markdown into one tree for MkDocs (Read the Docs).

MkDocs builds from a single docs directory, but the documentation here lives
beside the code it describes: the root README, ``packages/*/README.md``,
``packages/*/docs/*.md`` and a few package-level guides. This copies the tracked
Markdown into ``.docs_src/`` (git-ignored) with the same relative layout, so links
between documents keep working, and the root README becomes ``index.md``.

Links that would break on the built site are rewritten:

* a link to a package directory goes to that directory's ``README.md``;
* a link to anything not published (source files, ``requirements.txt``, ...) goes
  to the file on GitHub.

Run from the repository root:  ``python tools/collect_docs.py [OUT_DIR]``
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/ak-234/HiP-CT-vascular-toolkit/blob/main/"
SKIP = ("/tests/", "/research_scripts/", "/native/")
LINK = re.compile(r"(\]\()([^)\s#]+)(#[^)\s]*)?(\))")


def tracked_markdown() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout.split()
    return [p for p in out if not any(s in f"/{p}" for s in SKIP)]


def rewrite(text: str, source: str, published: set[str]) -> str:
    here = PurePosixPath(source).parent

    def fix(match: re.Match) -> str:
        target, anchor = match.group(2), match.group(3) or ""
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("/"):
            return match.group(0)  # absolute URL, mailto:, or site-absolute
        resolved = PurePosixPath(*(here / target).parts)
        parts = []
        for part in resolved.parts:  # normalise .. without touching the filesystem
            if part == "..":
                if parts:
                    parts.pop()
            elif part != ".":
                parts.append(part)
        path = "/".join(parts)
        if path in published:
            return match.group(0)
        if (REPO / path).is_dir() and f"{path}/README.md" in published:
            new = target.rstrip("/") + "/README.md"
            return f"{match.group(1)}{new}{anchor}{match.group(4)}"
        if (REPO / path).exists():
            return f"{match.group(1)}{GITHUB}{path}{anchor}{match.group(4)}"
        return match.group(0)  # leave genuinely broken links for MkDocs to report

    return LINK.sub(fix, text)


def main(out_dir: str = ".docs_src") -> int:
    out = REPO / out_dir
    if out.exists():
        shutil.rmtree(out)
    files = tracked_markdown()
    published = set(files)
    for source in files:
        dest = out / ("index.md" if source == "README.md" else source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = (REPO / source).read_text(encoding="utf-8")
        dest.write_text(rewrite(text, source, published), encoding="utf-8")
    # The root README is published as index.md; keep links to it working.
    for path in out.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        depth = len(path.relative_to(out).parts) - 1
        fixed = text.replace("](" + "../" * depth + "README.md", "](" + "../" * depth + "index.md")
        if fixed != text:
            path.write_text(fixed, encoding="utf-8")
    print(f"collected {len(files)} Markdown files into {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
