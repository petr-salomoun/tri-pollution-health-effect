#!/usr/bin/env python3
"""
export.py — Publish public artifacts to the export/ directory.

The export/ directory is intended to be a GitHub repository clone.
This script copies (never deletes) all public-facing artifacts there:
  - README.md  (repo main page)
  - DETAILS.md (methodology)
  - requirements.txt
  - run_pipeline.py
  - export.py (this script)
  - pipeline/ (all .py files, no __pycache__)
  - tests/ (all .py files)
  - Figures referenced in README.md

Data files and the interactive map are NOT exported (data is freely
available from EPA/CDC/Census; map is too large).

Usage:
    python export.py [--export-dir <path>]

Default export dir: ./export
"""

import argparse
import re
import shutil
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent

# Files to copy verbatim to the export root
ROOT_FILES = [
    "README.md",
    "DETAILS.md",
    "HOWTO.md",
    "requirements.txt",
    "run_pipeline.py",
    "export.py",
]

# Directories to copy (Python files only, no __pycache__)
CODE_DIRS = [
    "pipeline",
    "tests",
]

# Extra individual figures to include beyond what README references
EXTRA_FIGURES: list[str] = []

# ── Helpers ──────────────────────────────────────────────────────────────────

def _figures_from_readme(readme: Path) -> list[Path]:
    """Return all figure paths referenced in the README (relative to repo root)."""
    text = readme.read_text()
    matches = re.findall(r'!\[.*?\]\((output/[^)]+\.png)\)', text)
    return [REPO_ROOT / m for m in matches]


def _copy(src: Path, dst: Path, label: str = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    rel_src = label or str(src.relative_to(REPO_ROOT))
    print(f"  copied  {rel_src}")


def _gitignore_content() -> str:
    return """\
# Python
__pycache__/
*.py[cod]
*.pyo
.eggs/
*.egg-info/
dist/
build/

# Data (not published — download from EPA/CDC/Census)
data/

# Large outputs
output/map.html
output/RESEARCH_REPORT.*
output/research/

# Environment
.env
.venv/
venv/
"""

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export public artifacts for publication.")
    parser.add_argument("--export-dir", default="export",
                        help="Target directory (default: ./export)")
    args = parser.parse_args()

    export_dir = REPO_ROOT / args.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Export directory: {export_dir}\n")

    # 1. Root documents & scripts
    print("=== Documents & scripts ===")
    for rel in ROOT_FILES:
        src = REPO_ROOT / rel
        if src.exists():
            _copy(src, export_dir / rel)
        else:
            print(f"  MISSING: {rel}")

    # 2. Code directories (Python only, no __pycache__)
    print("\n=== Code ===")
    for dirname in CODE_DIRS:
        src_dir = REPO_ROOT / dirname
        if not src_dir.exists():
            print(f"  MISSING dir: {dirname}/")
            continue
        for src in sorted(src_dir.rglob("*.py")):
            if "__pycache__" in src.parts:
                continue
            rel = src.relative_to(REPO_ROOT)
            _copy(src, export_dir / rel)

    # 3. Figures referenced in README
    print("\n=== Figures ===")
    readme_path = REPO_ROOT / "README.md"
    figures = _figures_from_readme(readme_path)

    for extra in EXTRA_FIGURES:
        p = REPO_ROOT / extra
        if p not in figures:
            figures.append(p)

    for src in figures:
        if src.exists():
            rel = src.relative_to(REPO_ROOT)
            _copy(src, export_dir / rel)
        else:
            print(f"  MISSING: {src.relative_to(REPO_ROOT)}")

    # 4. Write .gitignore if it doesn't exist
    gitignore_path = export_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(_gitignore_content())
        print(f"\n  created .gitignore")

    n_code = sum(
        1 for d in CODE_DIRS
        for f in (REPO_ROOT / d).rglob("*.py")
        if "__pycache__" not in f.parts
    )
    print(f"\nDone. {len(ROOT_FILES)} root file(s) + {n_code} code file(s) + "
          f"{len(figures)} figure(s) exported to {export_dir}/")
    print("\nNext steps:")
    print("  cd export/")
    print("  git add -A && git commit -m 'publish: update analysis'")
    print("  git push")


if __name__ == "__main__":
    main()
