#!/usr/bin/env python3
r"""
Automated XeLaTeX Compilation & Log QA Inspector (Phase 4).

Compiles individual leaf monograph `.tex` files and/or `./master.tex` using:
    latexmk -xelatex -interaction=nonstopmode -halt-on-error

Parses generated `.log` files for LaTeX/TikZ/PGF errors, missing assets,
undefined references, and `Overfull \hbox` / `Overfull \vbox` warnings.

Usage examples:
    .venv/bin/python scripts/compile_qa.py --file master.tex
    .venv/bin/python scripts/compile_qa.py --domain 01_mathematical_foundations
    .venv/bin/python scripts/compile_qa.py --all
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Optional

ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^!\s+(?!\\(?:OML|OT1|OMS|OMX|TU)/).*"),
    re.compile(r"LaTeX Error:.*"),
    re.compile(r"Package\s+\S+\s+Error:.*"),
    re.compile(r"Undefined control sequence.*"),
    re.compile(r"Emergency stop\..*"),
    re.compile(r"Runaway argument\?"),
    re.compile(r"File\s+[`'].+?['`]\s+not found.*"),
]

OVERFULL_HBOX_RE = re.compile(
    r"Overfull \\hbox\s+\((?P<pt>[0-9]+(?:\.[0-9]+)?)pt too wide\)\s*(?P<loc>.*)"
)
OVERFULL_VBOX_RE = re.compile(
    r"Overfull \\vbox\s+\((?P<pt>[0-9]+(?:\.[0-9]+)?)pt too high\)\s*(?P<loc>.*)"
)
WARNING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"LaTeX Warning: Reference .* undefined.*"),
    re.compile(r"LaTeX Warning: There were undefined references\."),
    re.compile(r"Package hyperref Warning:.*"),
    re.compile(r"Package natbib Warning:.*"),
]


@dataclass(frozen=True)
class OverfullIssue:
    """Represents a single Overfull hbox/vbox warning extracted from a .log file."""

    kind: str
    overflow_pt: float
    location: str


@dataclass(frozen=True)
class CompileResult:
    """Compilation and QA inspection result for a single .tex target."""

    tex_path: str
    pdf_path: str
    exit_code: int
    passed: bool
    errors: list[str]
    overfull_hbox: list[OverfullIssue]
    overfull_vbox: list[OverfullIssue]
    other_warnings: list[str]
    max_overfull_hbox_pt: float


def _parse_log_file(log_path: Path) -> tuple[list[str], list[OverfullIssue], list[OverfullIssue], list[str]]:
    """Extract errors, overfull warnings, and undefined reference warnings from a TeX log."""
    if not log_path.is_file():
        return [f"Missing log file: {log_path}"], [], [], []

    raw_text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = raw_text.splitlines()

    errors: list[str] = []
    hbox_issues: list[OverfullIssue] = []
    vbox_issues: list[OverfullIssue] = []
    other_warnings: list[str] = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        for pat in ERROR_PATTERNS:
            if pat.search(line):
                ctx = stripped
                if idx + 1 < len(lines) and lines[idx + 1].strip():
                    ctx = f"{stripped} | {lines[idx + 1].strip()}"
                errors.append(ctx)
                break

        m_hb = OVERFULL_HBOX_RE.search(line)
        if m_hb:
            hbox_issues.append(
                OverfullIssue(
                    kind="hbox",
                    overflow_pt=float(m_hb.group("pt")),
                    location=m_hb.group("loc").strip(),
                )
            )
            continue

        m_vb = OVERFULL_VBOX_RE.search(line)
        if m_vb:
            vbox_issues.append(
                OverfullIssue(
                    kind="vbox",
                    overflow_pt=float(m_vb.group("pt")),
                    location=m_vb.group("loc").strip(),
                )
            )
            continue

        for wpat in WARNING_PATTERNS:
            if wpat.search(line):
                other_warnings.append(stripped)
                break

    return errors, hbox_issues, vbox_issues, other_warnings


def compile_tex_target(
    tex_path: Path,
    max_overfull_pt: float = 0.0,
    timeout_sec: int = 300,
    force: bool = False,
) -> CompileResult:
    """
    Compile one `.tex` target in its parent directory via `latexmk -xelatex`.

    Passes strictly when exit_code == 0, PDF exists, errors list is empty, and
    no `Overfull \\hbox` exceeds `max_overfull_pt`.
    """
    abs_tex = tex_path.resolve()
    if not abs_tex.is_file():
        return CompileResult(
            tex_path=str(tex_path),
            pdf_path="",
            exit_code=-1,
            passed=False,
            errors=[f"Source file not found: {abs_tex}"],
            overfull_hbox=[],
            overfull_vbox=[],
            other_warnings=[],
            max_overfull_hbox_pt=0.0,
        )

    cmd = [
        "latexmk",
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
    ]
    if force or abs_tex.name in ("master.tex", "atlas.tex"):
        cmd.append("-g")
    cmd.append(abs_tex.name)
    proc = subprocess.run(
        cmd,
        cwd=str(abs_tex.parent),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )

    log_path = abs_tex.with_suffix(".log")
    pdf_path = abs_tex.with_suffix(".pdf")
    errors, hbox_issues, vbox_issues, other_warnings = _parse_log_file(log_path)

    if abs_tex.name.endswith("_cheatsheet.tex") and pdf_path.is_file():
        try:
            import pymupdf

            with pymupdf.open(pdf_path) as doc:
                if len(doc) != 2:
                    errors.append(
                        f"Cheatsheet must be exactly 2 pages (Mindmap + Cheatsheet), got {len(doc)} pages."
                    )
        except Exception as exc:
            errors.append(f"Failed to inspect PDF page count: {exc}")
    if proc.returncode != 0 and not errors:
        tail = "\n".join(proc.stdout.splitlines()[-15:] + proc.stderr.splitlines()[-10:]).strip()
        errors.append(f"latexmk exited {proc.returncode}: {tail}")
    if not pdf_path.is_file():
        errors.append(f"Expected PDF output missing: {pdf_path}")

    max_hb = max((i.overflow_pt for i in hbox_issues), default=0.0)
    has_bad_overfull = any(i.overflow_pt > max_overfull_pt for i in hbox_issues)
    passed = proc.returncode == 0 and not errors and not has_bad_overfull

    return CompileResult(
        tex_path=str(tex_path),
        pdf_path=str(pdf_path) if pdf_path.is_file() else "",
        exit_code=int(proc.returncode),
        passed=passed,
        errors=errors,
        overfull_hbox=hbox_issues,
        overfull_vbox=vbox_issues,
        other_warnings=other_warnings,
        max_overfull_hbox_pt=round(max_hb, 3),
    )


def _gather_targets(
    taxonomy_path: Path,
    file_arg: Optional[Path],
    domain_id: Optional[str],
    compile_all: bool,
    cheatsheets_only: bool = False,
) -> list[Path]:
    """Resolve list of `.tex` files to compile from CLI arguments."""
    if file_arg:
        return [file_arg]

    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")
    with taxonomy_path.open("r", encoding="utf-8") as f:
        tree = json.load(f)

    targets: list[Path] = []
    for dom in tree.get("domains", []):
        if domain_id and dom.get("id") != domain_id:
            continue
        for pt in dom.get("parent_topics", []):
            for st in pt.get("sub_topics", []):
                for lf in st.get("leaves", []):
                    p = Path(lf["tex_path"])
                    cs = p.with_name(f"{p.stem}_cheatsheet.tex")
                    if cheatsheets_only:
                        if cs.is_file():
                            targets.append(cs)
                    else:
                        if p.is_file():
                            targets.append(p)
                        if compile_all and cs.is_file():
                            targets.append(cs)

    if compile_all:
        if Path("atlas.tex").is_file():
            targets.append(Path("atlas.tex"))
        if Path("master.tex").is_file():
            targets.append(Path("master.tex"))
    return targets


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compile LaTeX monograph files and inspect logs.")
    p.add_argument("--file", type=Path, default=None, help="Single .tex file to compile")
    p.add_argument("--domain", type=str, default=None, help="Compile all existing leaf .tex files in a domain")
    p.add_argument("--cheatsheets", action="store_true", help="Compile existing *_cheatsheet.tex files (optionally filtered by --domain)")
    p.add_argument("--all", action="store_true", help="Compile all existing leaf .tex, *_cheatsheet.tex, atlas.tex, and master.tex")
    p.add_argument("--taxonomy", type=Path, default=Path("data/taxonomy_tree.json"))
    p.add_argument(
        "--max-overfull-pt",
        type=float,
        default=0.0,
        help="Maximum tolerated Overfull \\hbox in pt (default: 0.0, strict zero tolerance)",
    )
    p.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    targets = _gather_targets(
        args.taxonomy,
        args.file,
        args.domain,
        args.all,
        cheatsheets_only=args.cheatsheets,
    )
    if not targets:
        print("No existing .tex target files found.", file=sys.stderr)
        return 1

    results = [compile_tex_target(t, max_overfull_pt=args.max_overfull_pt) for t in targets]
    passed_cnt = sum(1 for r in results if r.passed)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"[{status}] {r.tex_path} | exit={r.exit_code} | errors={len(r.errors)} | "
            f"overfull_hbox={len(r.overfull_hbox)} (max={r.max_overfull_hbox_pt}pt)"
        )
        for err in r.errors[:5]:
            print(f"    ERR : {err}")
        for hb in r.overfull_hbox[:5]:
            print(f"    HBOX: {hb.overflow_pt}pt {hb.location}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "targets": len(results),
            "passed": passed_cnt,
            "failed": len(results) - passed_cnt,
            "results": [asdict(r) for r in results],
        }
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return 0 if passed_cnt == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
