#!/usr/bin/env python3
"""Cleanup Utility for LaTeX Build Artifacts and Compilation Intermediates.

This script recursively scans the project workspace and removes auxiliary
LaTeX build artifacts (*.aux, *.log, *.out, *.toc, *.fls, *.fdb_latexmk, *.xdv,
*.synctex.gz, etc.) while preserving source code, extracted figures, and final
PDF documents.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Set, Tuple

# Standard LaTeX compilation auxiliary file extensions to remove
AUX_EXTENSIONS: Set[str] = {
    ".aux",
    ".log",
    ".out",
    ".toc",
    ".fls",
    ".fdb_latexmk",
    ".xdv",
    ".synctex.gz",
    ".bbl",
    ".blg",
    ".nav",
    ".snm",
    ".vrb",
    ".lof",
    ".lot",
    ".run.xml",
    ".bcf",
}

# Directories that must never be scanned or cleaned
EXCLUDE_DIRS: Set[str] = {
    ".git",
    ".venv",
    "books_monographs",
    "assets",
}


def find_auxiliary_files(root: Path) -> List[Path]:
    """Recursively discover all LaTeX auxiliary files within the workspace.

    Args:
        root: The root path of the repository workspace.

    Returns:
        List of Path objects representing auxiliary files to remove.
    """
    candidates: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place to avoid descending into them
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        current_path = Path(dirpath)
        for fname in filenames:
            file_path = current_path / fname
            # Check for multi-dot extension such as .synctex.gz or normal suffix
            if fname.endswith(".synctex.gz"):
                candidates.append(file_path)
            elif file_path.suffix.lower() in AUX_EXTENSIONS:
                candidates.append(file_path)

    return sorted(candidates)


def clean_files(files: Sequence[Path], dry_run: bool = False) -> Tuple[int, int]:
    """Delete the specified list of files and return count and freed bytes.

    Args:
        files: Sequence of Path objects to be deleted.
        dry_run: When True, simulates deletion without modifying filesystem.

    Returns:
        Tuple of (deleted_file_count, freed_bytes).
    """
    deleted_count = 0
    total_bytes = 0

    for f in files:
        try:
            size = f.stat().st_size
        except OSError:
            size = 0

        if not dry_run:
            try:
                f.unlink(missing_ok=True)
                deleted_count += 1
                total_bytes += size
            except OSError as err:
                print(f"[WARN] Failed to delete {f}: {err}")
        else:
            deleted_count += 1
            total_bytes += size

    return deleted_count, total_bytes


def main() -> None:
    """CLI entrypoint for cleaning auxiliary LaTeX build artifacts."""
    parser = argparse.ArgumentParser(
        description="Clean auxiliary LaTeX build files from the workspace."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root workspace path (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without deleting them",
    )
    args = parser.parse_args()

    root_dir = args.root.resolve()
    print(f"Scanning for auxiliary files in: {root_dir}")
    aux_files = find_auxiliary_files(root_dir)
    print(f"Discovered {len(aux_files)} auxiliary file(s).")

    if not aux_files:
        print("Workspace is already clean.")
        return

    count, total_bytes = clean_files(aux_files, dry_run=args.dry_run)
    mb_freed = total_bytes / (1024 * 1024)

    action_label = "Would remove" if args.dry_run else "Successfully removed"
    print(f"{action_label} {count} file(s) ({mb_freed:.2f} MB freed).")


if __name__ == "__main__":
    main()
