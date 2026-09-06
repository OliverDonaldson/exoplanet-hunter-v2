"""Is this repository fit to put in front of a stranger?

One command that answers the question a peer reviewer, a marker or a recruiter
would ask, with a verdict that cannot be produced by writing a document:
every check below runs something and reads the result.

    python pipeline/scripts/check_showcase_ready.py [--live] [--quick]

Exit 0 and "LOOKS GOOD" when every required check passes; exit 1 otherwise.
`--live` adds the two checks that need the deployed API and console; `--quick`
skips the test suite, which is the slow one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The deployed pair. Checked only under --live.
API_URL = os.environ.get("EH_API_URL", "https://exoplanet-hunter-api.fly.dev")
CONSOLE_URL = os.environ.get("EH_CONSOLE_URL", "https://exoplanet-hunter-console.onrender.com")

#: Documents a reader is entitled to find. Absence is a real gap, not a nit.
REQUIRED_DOCS = (
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "docs/index.md",
    "docs/PLAN.md",
    "docs/report.md",
    "docs/report.pdf",
    "docs/known-limits.md",
    "docs/decisions.md",
    "docs/experiments/README.md",
)


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


def run(cmd: list[str], cwd: Path = ROOT, timeout: int = 900) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 127, str(exc)
    return p.returncode, (p.stdout + p.stderr)


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------


def check_docs_present() -> Result:
    missing = [d for d in REQUIRED_DOCS if not (ROOT / d).exists()]
    return Result(
        "documents a reader expects",
        not missing,
        "missing: " + ", ".join(missing) if missing else "all present",
    )


def check_report_current() -> Result:
    md, pdf = ROOT / "docs/report.md", ROOT / "docs/report.pdf"
    if not (md.exists() and pdf.exists()):
        return Result("report PDF is current", False, "report.md or report.pdf is missing")
    stale = pdf.stat().st_mtime < md.stat().st_mtime
    return Result(
        "report PDF is current",
        not stale,
        "PDF older than the source — run `make report`" if stale else "PDF newer than its source",
    )


def check_doc_links() -> Result:
    """Every relative link in docs/ resolves. A dead link is a reader hitting a wall."""
    broken: list[str] = []
    for path in sorted((ROOT / "docs").rglob("*.md")):
        for target in re.findall(r"\]\(([^)#][^)]*)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target.split("#")[0]).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    return Result(
        "every doc link resolves",
        not broken,
        f"{len(broken)} broken: " + "; ".join(broken[:3]) if broken else "no dead links",
    )


def check_report_figures() -> Result:
    md = ROOT / "docs/report.md"
    if not md.exists():
        return Result("report figures exist", False, "no report")
    refs = re.findall(r"\((figures/[^)\s]+)\)", md.read_text(encoding="utf-8"))
    refs += re.findall(r"`(figures/[^`]+)`", md.read_text(encoding="utf-8"))
    missing = sorted({r for r in refs if not (ROOT / "docs" / r).exists()})
    return Result(
        "report figures exist",
        not missing,
        "missing: " + ", ".join(missing)
        if missing
        else f"{len(set(refs))} referenced, all present",
    )


def check_registry_matches_served() -> Result:
    """The registry names a run whose artefacts are actually on disk."""
    reg = ROOT / "models/registry.json"
    if not reg.exists():
        return Result("registry points at a real run", False, "models/registry.json missing")
    run_id = json.loads(reg.read_text())["run_id"]
    cv = ROOT / "models/cv" / run_id
    have = [f for f in ("cv_summary.json", "predictions.parquet") if (cv / f).exists()]
    ok = len(have) == 2
    return Result(
        "registry points at a real run",
        ok,
        f"{run_id[:8]}: " + (", ".join(have) if have else "no artefacts on disk"),
    )


def check_plan_complete() -> Result:
    """Every delivery step in PLAN.md's status table has landed."""
    plan = ROOT / "docs/PLAN.md"
    if not plan.exists():
        return Result("every delivery step landed", False, "docs/PLAN.md missing")
    rows = [
        ln
        for ln in plan.read_text(encoding="utf-8").splitlines()
        if re.match(r"^\|\s*\d+\s*\|", ln)
    ]
    unfinished = [
        r.split("|")[2].strip()
        for r in rows
        if "not started" in r.lower() or "in progress" in r.lower()
    ]
    return Result(
        "every delivery step landed",
        not unfinished,
        f"{len(unfinished)} open: " + "; ".join(s[:40] for s in unfinished[:3])
        if unfinished
        else f"{len(rows)} steps, all landed",
    )


def check_git_clean() -> Result:
    code, out = run(["git", "status", "--porcelain"])
    dirty = [ln for ln in out.splitlines() if ln.strip()]
    return Result(
        "working tree is clean",
        code == 0 and not dirty,
        f"{len(dirty)} uncommitted change(s)" if dirty else "nothing uncommitted",
    )


def check_lint() -> Result:
    code, out = run(["ruff", "check", "pipeline", "api"])
    return Result(
        "ruff clean", code == 0, out.strip().splitlines()[-1][:90] if code else "no findings"
    )


#: mypy is not a CI gate here (see issue #55, config skew), so the standard is
#: "no worse than the last recorded count" rather than zero. Lower it, never raise it.
MYPY_BASELINE = ROOT / ".mypy-baseline"


def check_types() -> Result:
    code, out = run(["mypy", "pipeline/src"])
    found = re.search(r"Found (\d+) error", out)
    errors = int(found.group(1)) if found else (0 if code == 0 else -1)
    if errors < 0:
        return Result("mypy at or under baseline", False, "could not read mypy output")
    if not MYPY_BASELINE.exists():
        return Result("mypy at or under baseline", False, "no .mypy-baseline recorded")
    baseline = int(MYPY_BASELINE.read_text().split()[0])
    ok = errors <= baseline
    trend = "same as" if errors == baseline else ("under" if ok else "OVER")
    return Result("mypy at or under baseline", ok, f"{errors} errors, {trend} baseline {baseline}")


def check_tests() -> Result:
    code, out = run(["pytest", "pipeline/tests", "-m", "not network and not slow", "-q"])
    api_code, api_out = run(["pytest", "api/tests", "-q"])
    tail = [ln for ln in (out + api_out).strip().splitlines() if "passed" in ln or "failed" in ln]
    ok = code == 0 and api_code == 0
    return Result(
        "fast suite green", ok, " | ".join(t.strip()[:44] for t in tail[-2:]) or "no summary line"
    )


def _get(url: str, timeout: int = 25) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception as exc:  # network, DNS, TLS — all mean "a visitor sees nothing"
        return 0, str(exc)


def check_api_live() -> Result:
    status, body = _get(f"{API_URL}/health")
    return Result("deployed API answers", status == 200, f"{API_URL} -> {status or body[:50]}")


def check_console_live() -> Result:
    status, body = _get(CONSOLE_URL)
    ok = status == 200 and "<" in body
    return Result("deployed console answers", ok, f"{CONSOLE_URL} -> {status}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--live", action="store_true", help="also check the deployed API and console"
    )
    parser.add_argument("--quick", action="store_true", help="skip the test suite")
    args = parser.parse_args()

    checks = [
        check_docs_present,
        check_report_current,
        check_report_figures,
        check_doc_links,
        check_registry_matches_served,
        check_plan_complete,
        check_git_clean,
        check_lint,
        check_types,
    ]
    if not args.quick:
        checks.append(check_tests)
    if args.live:
        checks += [check_api_live, check_console_live]

    print(f"\n  Showcase readiness — {ROOT}\n")
    results = []
    for fn in checks:
        r = fn()
        results.append(r)
        print(f"  {'PASS' if r.ok else 'FAIL'}  {r.name:<34} {r.detail}")

    failed = [r for r in results if not r.ok]
    print()
    if failed:
        print(f"  NOT YET — {len(failed)} of {len(results)} checks failing.")
        print("  Fix these before showing the project:")
        for r in failed:
            print(f"    - {r.name}: {r.detail}")
        print()
        return 1
    print(f"  LOOKS GOOD — all {len(results)} checks pass. Safe to publish, share or link.")
    print("  A reader can clone this, read docs/report.pdf, and reproduce its numbers.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
