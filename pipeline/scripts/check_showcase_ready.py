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
import ast
import io
import json
import os
import re
import subprocess
import sys
import tokenize
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: The deployed pair. Checked only under --live.
API_URL = os.environ.get("EH_API_URL", "https://exoplanet-hunter-api.fly.dev")
CONSOLE_URL = os.environ.get("EH_CONSOLE_URL", "https://exoplanet-hunter-console.onrender.com")

#: What the console fetches on every load, so a deployment missing any one of
#: them puts an error string in a panel in front of a visitor. Checked rather
#: than assumed because this list drifted once: the probe asked for /health,
#: which no version of the API has ever served, so --live failed identically
#: whether the deployment was current or five commits behind. /score is left
#: out: it is on demand and can take minutes.
CONSOLE_ENDPOINTS = (
    "/healthz",
    "/model",
    "/model/training-history",
    "/reliability",
    "/runs?limit=8",
    "/candidates?limit=1&sort_by=prob_mean&order=desc",
)

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


def _strip_code_fences(text: str) -> str:
    """Drop fenced blocks. A link inside one is quoted text — often drafted prose
    for another file, where its relative path is correct and ours is not."""
    return re.sub(r"^```.*?^```", "", text, flags=re.DOTALL | re.MULTILINE)


def check_doc_links() -> Result:
    """Every relative link in docs/ resolves. A dead link is a reader hitting a wall."""
    broken: list[str] = []
    for path in sorted((ROOT / "docs").rglob("*.md")):
        body = _strip_code_fences(path.read_text(encoding="utf-8"))
        for target in re.findall(r"\]\(([^)#][^)]*)\)", body):
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
    # A fresh clone or worktree carries the DVC pointers but not the bytes —
    # the same thing a stranger sees. Name the fix, not just the symptom.
    detail = ", ".join(have) if have else "no artefacts on disk — run `make data-pull`"
    return Result("registry points at a real run", ok, f"{run_id[:8]}: {detail}")


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


#: PLAN.md section 2 step 8's exit criteria. Measured here rather than recorded
#: once, because the pass landed a handful of lines under the bar and one
#: enthusiastic docstring erases that.
_COMMENT_SHARE_MAX = 25.0
_MODULE_DOCSTRING_MAX = 15
_TEST_NAME_MAX = 60


def _prose_census(root: Path) -> tuple[int, int, list[tuple[str, int]]]:
    """Physical lines, docstring-plus-comment lines, and over-long module docstrings."""
    total = prose = 0
    over: list[tuple[str, int]] = []
    for path in sorted(root.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        total += len(src.splitlines())
        marked: set[int] = set()
        for node in ast.walk(ast.parse(src)):
            if not isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                continue
            if ast.get_docstring(node, clean=False) is None:
                continue
            span = range(node.body[0].lineno, node.body[0].end_lineno + 1)
            marked.update(span)
            if isinstance(node, ast.Module) and len(span) > _MODULE_DOCSTRING_MAX:
                over.append((str(path.relative_to(ROOT)), len(span)))
        prose += len(marked)
        prose += sum(
            1
            for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type == tokenize.COMMENT
        )
    return total, prose, over


def check_comment_share() -> Result:
    total, prose, _ = _prose_census(ROOT / "pipeline" / "src")
    share = prose / total * 100
    ok = share < _COMMENT_SHARE_MAX
    headroom = int(total * _COMMENT_SHARE_MAX / 100) - prose
    return Result(
        "comment share under 25%",
        ok,
        f"{share:.2f}% of {total:,} lines — {headroom} lines of headroom"
        if ok
        else f"{share:.2f}%, over by {-headroom} lines",
    )


def check_module_docstrings() -> Result:
    _, _, over = _prose_census(ROOT / "pipeline" / "src")
    worst = ", ".join(f"{p} ({n})" for p, n in sorted(over, key=lambda x: -x[1])[:3])
    return Result(
        "no module docstring over 15 lines",
        not over,
        f"{len(over)} over: {worst}" if over else "all 15 lines or fewer",
    )


def check_test_names() -> Result:
    long: list[str] = []
    for root in (ROOT / "pipeline" / "tests", ROOT / "api" / "tests"):
        for path in sorted(root.rglob("test_*.py")):
            for name in re.findall(
                r"^\s*(?:async )?def (test_\w+)", path.read_text(encoding="utf-8"), re.M
            ):
                if len(name) > _TEST_NAME_MAX:
                    long.append(name)
    return Result(
        "test names under 60 characters",
        not long,
        f"{len(long)} over, longest {max((len(n) for n in long), default=0)}"
        if long
        else "all under 60",
    )


#: The console is the project's primary artefact — the thing a visitor actually
#: opens. A gate that passes while it cannot be built is checking the wrong
#: deliverable, which this one did until 2026-09-10.
def check_console_builds() -> Result:
    """The static console builds into the single file Render serves."""
    frontend = ROOT / "frontend"
    if not (frontend / "node_modules").exists():
        return Result("console builds", False, "no node_modules — run `npm install` in frontend/")
    code, out = run(["python3", "design-console/build.py"], cwd=frontend, timeout=300)
    built = frontend / "design-console" / "dist" / "index.html"
    size = built.stat().st_size if built.exists() else 0
    ok = code == 0 and size > 0
    if ok:
        return Result("console builds", True, f"dist/index.html, {size / 1024:.0f} kB")
    tail = next((ln for ln in reversed(out.strip().splitlines()) if ln.strip()), "no output")
    return Result("console builds", False, tail[:90])


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


def _get(url: str, timeout: int = 25, retry_slow: bool = False) -> tuple[int, str]:
    """`retry_slow` retries once after a timeout and only after a timeout, which
    is the same rule app.api.js::probeApi follows and for the same reason: the
    API suspends when idle, so the first request of the day pays a machine
    resume and a TensorFlow warm. A refused connection or a bad host does not
    get better on a second go."""
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, r.read(4096).decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, ""
        except TimeoutError as exc:
            if retry_slow and attempt == 0:
                continue
            return 0, str(exc) or "timed out"
        except Exception as exc:  # network, DNS, TLS — all mean "a visitor sees nothing"
            return 0, str(exc)
    return 0, "timed out twice"


def check_api_live() -> Result:
    missing = []
    for i, path in enumerate(CONSOLE_ENDPOINTS):
        status, body = _get(f"{API_URL}{path}", retry_slow=i == 0)
        if status != 200:
            missing.append(f"{path} -> {status or body[:40]}")
    detail = ", ".join(missing) or f"{len(CONSOLE_ENDPOINTS)} console endpoints, all 200"
    return Result("deployed API answers", not missing, detail)


def check_console_live() -> Result:
    status, body = _get(CONSOLE_URL)
    ok = status == 200 and "<" in body
    return Result("deployed console answers", ok, f"{CONSOLE_URL} -> {status}")


def check_link_preview() -> Result:
    """A shared link is how most people meet this project, and a card with no
    image is the same grey box as a dead link. The tag is only half of it: the
    URL it names has to answer, which is a separate deploy from the HTML."""
    status, body = _get(CONSOLE_URL)
    if status != 200:
        return Result("shared link previews", False, f"{CONSOLE_URL} -> {status}")
    match = re.search(r'<meta property="og:image" content="([^"]+)"', body)
    if not match:
        return Result("shared link previews", False, "no og:image tag in the served head")
    img_status, _ = _get(match.group(1))
    return Result(
        "shared link previews",
        img_status == 200,
        f"og:image -> {img_status}" if img_status != 200 else match.group(1).rsplit("/", 1)[-1],
    )


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
        check_console_builds,
        check_comment_share,
        check_module_docstrings,
        check_test_names,
        check_git_clean,
        check_lint,
        check_types,
    ]
    if not args.quick:
        checks.append(check_tests)
    if args.live:
        checks += [check_api_live, check_console_live, check_link_preview]

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
