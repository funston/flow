"""`/flow` — a gated issue-to-PR workflow for unfamiliar repositories.

    onboard -> issue -> repro -> fix -> verify -> review -> pr

This module is surface-agnostic and stdlib-only. The CLI shell and the ACP
bridge both call `run_flow()` and get back a `FlowResult`:

  * `message`       markdown to show the user (always set)
  * `agent_prompt`  if set, the surface runs it through the reasoning model
  * `state`         the persisted `FlowState`, for rendering a checklist/plan

The split is deliberate. Everything that can be decided without a model is
decided here, deterministically: which test command to run, whether the
regression test is actually red, whether the suite regressed against the
baseline, whether the diff changed after it was verified. The model is only
asked to do the parts that need judgment (trace a root cause, write a test,
write a fix, review a diff), and it cannot advance a phase by claiming
success — phases advance on observed exit codes and diff hashes.

State lives inside the git directory (`.git/bach-flow.json`) so it is
per-clone and can never be committed by accident.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

PHASES: list[tuple[str, str]] = [
    ("onboard", "Map the repo and record a test baseline"),
    ("issue", "Capture the issue, branch, trace the root cause"),
    ("repro", "Regression test that fails for the right reason"),
    ("fix", "Smallest correct change"),
    ("verify", "Regression test green, suite no worse than baseline"),
    ("review", "Skeptical review of the diff against the issue"),
    ("pr", "Commit, push, open the pull request"),
]
PHASE_NAMES = [name for name, _ in PHASES]

_STATE_FILE = "bach-flow.json"
_PR_DRAFT = "PR-DRAFT.md"         # the pull request description, in the project directory
_SCOPE_FILE = "bach-flow-scope"   # the directory chosen with `/flow scope`
_SCOPE_DIR = "bach-flow"          # per-directory state lives in here
_MAX_DIFF_CHARS = 12_000
_INSTRUCTION_FILES = (
    "TASK.md", "README.md", "README.rst", "CONTRIBUTING.md",
    "CLAUDE.md", "AGENTS.md",
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class SuiteRun:
    command: str
    returncode: int
    passed: Optional[int] = None
    failed: Optional[int] = None
    errors: Optional[int] = None
    summary: str = ""
    seconds: float = 0.0
    # Ids of the tests that failed or errored, when the runner names them.
    failing: list = field(default_factory=list)

    @property
    def bad(self) -> Optional[int]:
        """Failures + errors, or None when the output could not be parsed."""
        if self.failed is None and self.errors is None:
            return None
        return (self.failed or 0) + (self.errors or 0)


@dataclass
class FlowState:
    test_command: str = ""
    base_branch: str = ""
    baseline: Optional[dict] = None
    issue: str = ""
    issue_title: str = ""
    # Tracker reference and link when the issue came from Linear (ENG-123).
    issue_ref: str = ""
    issue_url: str = ""
    # "bug" (the default) or "feature". A feature is built from a written
    # spec: step 2 plans instead of tracing a root cause, and step 3 writes
    # acceptance tests from the spec's acceptance criteria.
    kind: str = "bug"
    plan: str = ""            # the recorded plan for a feature
    plan_posted: bool = False  # ...and whether it reached the tracker
    branch: str = ""
    repro_target: str = ""
    repro_requested: bool = False
    repro_red: Optional[dict] = None
    fix_requested: bool = False
    verified: Optional[dict] = None
    # Why the last `/flow verify` failed ("" when it passed or has not run).
    verify_failed: str = ""
    verified_diff: str = ""
    reviewed_diff: str = ""
    # What the reviewer said, kept so `/flow fix` can act on it (the model
    # has no memory between prompts), and whether that has been requested.
    review_findings: str = ""
    review_addressed: bool = False
    pr_drafted: bool = False         # `/flow pr` wrote the description; not opened yet
    author: str = ""        # provider that was asked for the fix
    reviewer: str = ""      # provider chosen with `/flow review --with`
    reviewed_by: str = ""   # who actually completed the last review
    pr_url: str = ""
    log: list[str] = field(default_factory=list)

    def done(self, phase: str) -> bool:
        return {
            "onboard": self.baseline is not None,
            "issue": bool(self.issue),
            "repro": self.repro_red is not None,
            "fix": self.fix_requested,
            "verify": self.verified is not None,
            "review": bool(self.reviewed_diff),
            "pr": bool(self.pr_url),
        }[phase]

    def current(self) -> str:
        """First phase that is not done yet ('pr' when everything is)."""
        for name in PHASE_NAMES:
            if not self.done(name):
                return name
        return "pr"


@dataclass
class FlowResult:
    message: str
    state: FlowState
    agent_prompt: Optional[str] = None
    ok: bool = True
    # Set only by `/flow review`. The surface runs REVIEW_PROMPT through a
    # separate read-only reviewer (bach.ops.critique), optionally on the
    # REVIEWER provider, then calls `record_review(REVIEW_FINGERPRINT)`.
    review_prompt: Optional[str] = None
    reviewer: str = ""
    review_fingerprint: str = ""


# ---------------------------------------------------------------------------
# Small process helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path, timeout: float = 60.0) -> tuple[int, str]:
    """Run CMD, return (returncode, combined output). Never raises."""
    # No bytecode caches from our runs: CPython validates a .pyc by
    # (mtime, size), so a same-size edit within a second of the previous
    # test run would otherwise be tested against stale bytecode — exactly
    # the "one-character fix" case, and a wrong red/green either way.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
            env=env,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]}: command not found"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s: {' '.join(cmd)}"


def _git(root: Path, *args: str, timeout: float = 60.0) -> tuple[int, str]:
    code, out = _run(["git", *args], root, timeout)
    return code, out.strip()


def _git_dir(root: Path) -> Optional[Path]:
    code, out = _git(root, "rev-parse", "--absolute-git-dir")
    return Path(out) if code == 0 and out else None


def _toplevel(root: Path) -> Optional[Path]:
    code, out = _git(root, "rev-parse", "--show-toplevel")
    return Path(out).resolve() if code == 0 and out else None


def scope_of(root: Path) -> str:
    """ROOT relative to the top of its repository: "" at the top, else a
    path such as `services/billing`. The workflow is scoped to that
    directory: its own saved state, its own tests, its own diff."""
    top = _toplevel(root)
    if top is None:
        return ""
    try:
        rel = root.resolve().relative_to(top)
    except ValueError:
        return ""
    return "" if str(rel) == "." else rel.as_posix()


def _scoped_file(root: Path, name: str) -> Optional[Path]:
    """Where a per-scope file lives inside the git directory: the plain
    name at the top of the repository, `bach-flow/<scope>.<name>` below it."""
    git_dir = _git_dir(root)
    if git_dir is None:
        return None
    scope = scope_of(root)
    if not scope:
        return git_dir / name
    return git_dir / _SCOPE_DIR / f"{scope.replace('/', '__')}.{name}"


def _state_path(root: Path) -> Path:
    path = _scoped_file(root, _STATE_FILE)
    return path if path is not None else root / ".bach" / _STATE_FILE


def _chosen_scope(top: Path) -> str:
    """The directory picked with `/flow scope`, or "" when none is."""
    git_dir = _git_dir(top)
    if git_dir is None:
        return ""
    try:
        chosen = (git_dir / _SCOPE_FILE).read_text().strip()
    except OSError:
        return ""
    return chosen if chosen and (top / chosen).is_dir() else ""


def project_root(project_path: str | Path) -> Path:
    """The directory the workflow operates on.

    Started inside a subdirectory of a repository: that subdirectory.
    Started at the top of a repository: the directory chosen with
    `/flow scope`, or the top itself when none is chosen. This is what
    lets one large repository hold many independent pieces of work.
    """
    root = Path(project_path).expanduser().resolve()
    top = _toplevel(root)
    if top is None or root != top:
        return root
    chosen = _chosen_scope(top)
    return top / chosen if chosen else root


def load_state(root: Path) -> FlowState:
    path = _state_path(root)
    if not path.exists():
        return FlowState()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return FlowState()
    known = {k: v for k, v in raw.items() if k in FlowState.__dataclass_fields__}
    return FlowState(**known)


def save_state(root: Path, state: FlowState) -> None:
    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2))


def _base_ref(root: Path, state: "FlowState") -> str:
    """What the work is measured against: the branch we started from, or HEAD
    when no separate branch was created."""
    _, current = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if state.base_branch and state.base_branch != current:
        if _git(root, "rev-parse", "--verify", "--quiet", state.base_branch)[0] == 0:
            return state.base_branch
    return "HEAD"


def review_base(project_path: str | Path) -> str:
    """The ref the current workflow's diff is measured against. For surfaces
    that need to fingerprint the diff themselves before recording a review."""
    root = project_root(project_path)
    return _base_ref(root, load_state(root))


def _is_bach_artifact(path: str) -> bool:
    """Files BACH itself drops in a project (Ranger stats, session
    transcripts). They are not part of anyone's change: they must not
    invalidate a verified diff, appear in a review, or land in a PR."""
    name = path.rsplit("/", 1)[-1]
    if path in (".ranger.json", ".ranger_stats", _PR_DRAFT):
        return True
    return path.startswith("session/") and name in ("_history.jsonl", "transcript.jsonl")


def _changed_paths(root: Path, base: str) -> tuple[list[str], list[str]]:
    """(paths that differ from BASE, untracked paths), minus BACH's own files."""
    # --relative: only files under ROOT, named relative to it. In a large
    # repository, changes elsewhere are not part of this piece of work.
    _, tracked = _git(root, "diff", base, "--name-only", "--relative")
    _, untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    return (
        sorted(p for p in tracked.splitlines() if p),
        sorted(p for p in untracked.splitlines() if p and not _is_bach_artifact(p)),
    )


def diff_fingerprint(root: Path, base: str = "HEAD") -> str:
    """Hash of the on-disk content of every file that differs from BASE.

    Used to detect "the code changed after it was verified/reviewed". It
    depends only on file contents, so committing or staging the same work
    does not change it.
    """
    tracked, untracked = _changed_paths(root, base)
    digest = hashlib.sha256()
    for name in sorted(set(tracked) | set(untracked)):
        digest.update(name.encode() + b"\0")
        try:
            digest.update((root / name).read_bytes())
        except OSError:
            digest.update(b"<deleted>")
    return digest.hexdigest()[:16]


def working_diff(root: Path, base: str = "HEAD") -> str:
    """Unified diff against BASE, including untracked files, without
    touching the index."""
    _, diff = _git(root, "diff", base, "--relative")
    parts = [diff] if diff else []
    for name in _changed_paths(root, base)[1]:
        _, new_file = _git(root, "diff", "--no-index", "--", "/dev/null", name)
        if new_file:
            parts.append(new_file)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Test command detection and execution
# ---------------------------------------------------------------------------


def detect_test_command(root: Path) -> str:
    """Best guess at the project's test command, or '' if unknown."""
    has_py_tests = any(
        (root / d).is_dir() and any((root / d).rglob("test_*.py"))
        for d in ("tests", "test")
    )
    py_markers = ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "setup.py")
    if has_py_tests or (root / "pytest.ini").exists():
        if has_py_tests or any((root / m).exists() for m in py_markers):
            return f"{shlex.quote(_project_python(root))} -m pytest -q"

    package_json = root / "package.json"
    if package_json.exists():
        try:
            scripts = json.loads(package_json.read_text()).get("scripts", {})
        except (OSError, json.JSONDecodeError):
            scripts = {}
        if "test" in scripts:
            return "npm test --silent"
    if (root / "Cargo.toml").exists():
        return "cargo test"
    if (root / "go.mod").exists():
        return "go test ./..."
    makefile = root / "Makefile"
    if makefile.exists() and re.search(r"^test:", makefile.read_text(), re.M):
        return "make test"
    return ""


def _project_python(root: Path) -> str:
    """Prefer the project's own virtualenv over the one BACH runs in."""
    for candidate in (".venv/bin/python", "venv/bin/python"):
        if (root / candidate).exists():
            return str(root / candidate)
    # No project virtualenv: use the Python the user's shell would run,
    # which is where the project's dependencies (torch, numpy, ...) live.
    # BACH's own interpreter only has BACH's dependencies, so it is the
    # last resort, not the default.
    bach_python = Path(sys.executable).resolve()
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found and Path(found).resolve() != bach_python:
            return found
    return sys.executable


_COUNT = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "errors": re.compile(r"(\d+) errors?"),
}


def parse_test_output(output: str) -> dict:
    """Pull pass/fail counts out of a test runner's output (pytest-shaped)."""
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    summary = ""
    for line in reversed(lines):
        if any(rx.search(line) for rx in _COUNT.values()) or "no tests ran" in line:
            summary = line.strip("= ")
            break
    if not summary and lines:
        summary = lines[-1][:200]
    counts: dict = {"summary": summary}
    for key, rx in _COUNT.items():
        match = rx.search(summary)
        counts[key] = int(match.group(1)) if match else None
    if any(counts[k] is not None for k in _COUNT):
        for key in _COUNT:
            counts[key] = counts[key] or 0
    return counts


def run_tests(
    root: Path, command: str, target: str = "", timeout: float = 900.0
) -> tuple[SuiteRun, str]:
    """Run the test command (optionally narrowed to TARGET)."""
    argv = shlex.split(command) + ([target] if target else [])
    started = time.monotonic()
    code, output = _run(argv, root, timeout)
    parsed = parse_test_output(output)
    run = SuiteRun(
        command=" ".join(shlex.quote(a) for a in argv),
        returncode=code,
        passed=parsed["passed"],
        failed=parsed["failed"],
        errors=parsed["errors"],
        summary=parsed["summary"],
        seconds=round(time.monotonic() - started, 2),
        failing=_failing_ids(output),
    )
    return run, output


def _tail(output: str, lines: int = 25) -> str:
    kept = output.strip().splitlines()[-lines:]
    return "```\n" + "\n".join(kept) + "\n```"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def checklist(state: FlowState) -> str:
    """Plain-text phase checklist: [x] done, [>] current, [ ] pending."""
    current = state.current()
    rows = []
    for name, description in PHASES:
        if state.done(name):
            mark = "[x]"
        elif name == current:
            mark = "[>]"
        else:
            mark = "[ ]"
        rows.append(f"{mark} {name:<8} {description}")
    return "\n".join(rows)


NEXT_MARK = " → next: "


def plan_entries(state: FlowState, root: Optional[Path] = None) -> list[tuple[str, str]]:
    """(content, status) pairs for an ACP plan update.

    The current phase carries the next command (`... → next: /flow fix`) so
    an editor that pins the plan always shows what to do, not just where
    we are.
    """
    current = state.current()
    command, _ = next_step(state, root)
    out = []
    for name, description in PHASES:
        if state.done(name):
            status = "completed"
        elif name == current:
            status = "in_progress"
        else:
            status = "pending"
        content = f"{name}: {description}"
        if name == current and not state.pr_url:
            content += f"{NEXT_MARK}{command}"
        out.append((content, status))
    return out


def next_step(state: FlowState, root: Optional[Path] = None) -> tuple[str, str]:
    """(command, where things stand) for the current point in the workflow.

    Several phases have two halves (ask the model, then check its work
    ourselves), so "which phase" alone does not say what to do next.
    """
    if root is not None:
        root = project_root(root)
    stale = bool(root is not None and state.verified and _stale(root, state))
    if state.baseline is None:
        return "/flow onboard", "Nothing recorded yet. Start by mapping the repo and running its tests."
    baseline = state.baseline.get("summary") or "recorded"
    if not state.issue:
        return "/flow issue", f"Baseline is {baseline}. No issue chosen yet."
    if state.repro_red is None:
        if state.repro_requested:
            return (
                "/flow repro <test id>",
                "A regression test was requested, but nobody has run it yet. Run "
                "it by its id (the model's reply names it): it has to fail.",
            )
        if state.kind == "feature":
            criteria = acceptance_criteria(state.issue)
            have = (
                f"The spec lists {len(criteria)} acceptance criteria."
                if criteria else
                "The spec has no acceptance criteria yet: add an `Acceptance "
                "criteria` list to the issue first, or the next step will refuse."
            )
            plan = "The plan is recorded" + (" and posted to the issue. " if state.plan_posted else ". ") \
                if state.plan else "A plan was requested; no code has changed. "
            return "/flow repro", plan + have + " Next: acceptance tests written from those criteria."
        return (
            "/flow repro",
            "The issue is captured and its cause traced, with no code changed. "
            "Check the explanation against the lines it cites, then ask for a "
            "failing test.",
        )
    red = state.repro_red.get("summary") or "fails"
    if not state.fix_requested:
        return (
            "/flow fix",
            f"The regression test `{state.repro_target}` fails as it should ({red}). "
            "The bug is proven; no fix has been asked for.",
        )
    if state.verified is None and state.verify_failed:
        return (
            "/flow fix",
            f"The last verify failed: {state.verify_failed}. The fix is "
            "incomplete or the diagnosis was wrong. Read the failure, decide "
            "which, then ask for another fix (it will be given the failure).",
        )
    if state.verified is None:
        return (
            "/flow verify",
            "A fix was requested. It is unproven until the tests are run: read "
            "the diff, then verify.",
        )
    if stale:
        return (
            "/flow verify",
            "The code changed after the last verify, so that result no longer "
            "counts. Verify again.",
        )
    if not state.reviewed_diff:
        return (
            "/flow review",
            f"Verified: {state.verified.get('summary') or 'tests green'}. "
            "The diff has not been reviewed.",
        )
    if state.review_findings and not state.review_addressed and not state.pr_url and not state.pr_drafted:
        return (
            "/flow fix",
            f"Reviewed by {state.reviewed_by or 'the reviewer'}, who reported "
            "findings. Decide which are real. `/flow fix` hands them to the "
            "model; add your direction to narrow it, e.g. `/flow fix only "
            "findings 1 and 3`. Then verify and review again. If none need "
            "action, go straight to `/flow pr`.",
        )
    if not state.pr_url and state.pr_drafted:
        return (
            "/flow pr --create",
            f"The pull request description is drafted in `{_PR_DRAFT}`; nothing is "
            "committed or pushed. Read it and correct anything that is wrong, "
            "then create the pull request. "
            "To stop here without opening one, pick the next issue with "
            "`/flow issue`.",
        )
    if not state.pr_url:
        return (
            "/flow pr",
            f"Verified and reviewed by {state.reviewed_by or 'the reviewer'}. "
            "Ready to draft the pull request.",
        )
    return "/flow issue", f"Pull request opened: {state.pr_url}. Pick the next issue."


def _blocked(state: FlowState, phase: str, why: str, hint: str) -> FlowResult:
    return FlowResult(
        message=(
            f"**/flow {phase} is blocked:** {why}\n\n{hint}\n\n"
            f"Use `/flow {phase} --force` to override.\n\n```\n{checklist(state)}\n```"
        ),
        state=state,
        ok=False,
    )


# ---------------------------------------------------------------------------
# Prompts for the phases that need the reasoning model
# ---------------------------------------------------------------------------


def _recon_prompt(state: FlowState, found: list[str]) -> str:
    base = state.baseline or {}
    return f"""I am new to this repository.
Inspect it without modifying anything.

Already established (do not re-derive):
- Test command: {state.test_command or "unknown"}
- Baseline: {base.get("summary") or "not recorded"}
- Instruction files present: {", ".join(found) or "none found"}

Report, briefly:
1. Language, framework, and build system.
2. The main components and entry points, with file paths.
3. What the instruction files ask of me.
4. How to run a single test.
5. Test and error-handling conventions I should follow.

Keep it to a map I can read in one minute. Say so where you are unsure."""


def _investigate_prompt(state: FlowState) -> str:
    return f"""Investigate this issue. Do not modify any code.

ISSUE:
{state.issue}

Trace the execution path from the public entry point to the faulty behaviour
and report:
1. The functions and files involved, in call order.
2. Expected versus actual behaviour.
3. The most likely root cause, quoting the lines that support it.
4. Whether that cause explains every symptom in the issue; list any it does not.
5. Alternative hypotheses worth ruling out.
6. The smallest test that would reproduce the failure.

Mark each statement as observed in the code or as a hypothesis."""


_CRITERIA_HEADING = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:\*\*|__)?\s*acceptance criteria\b", re.IGNORECASE
)
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s*)?(\S.*)$")
_ANY_HEADING = re.compile(r"^\s{0,3}(?:#{1,6}\s+\S|(?:\*\*|__).+(?:\*\*|__)\s*:?\s*$)")


def acceptance_criteria(text: str) -> list[str]:
    """The list items under an `Acceptance criteria` heading in a spec.

    This is what makes a feature request testable: each item becomes an
    acceptance test. No heading, or a heading with no list, gives [].
    """
    lines = text.splitlines()
    for at, line in enumerate(lines):
        if not _CRITERIA_HEADING.match(line):
            continue
        items: list[str] = []
        for later in lines[at + 1:]:
            item = _LIST_ITEM.match(later)
            if item:
                items.append(item.group(1).strip())
            elif not later.strip():
                continue
            elif _ANY_HEADING.match(later):
                break
            elif items and not later[:1].isspace():
                break  # prose after the list: the list is over
        if items:
            return items
    return []


def _plan_prompt(state: FlowState) -> str:
    criteria = acceptance_criteria(state.issue)
    listed = (
        "\n".join(f"  {n}. {c}" for n, c in enumerate(criteria, 1))
        if criteria else
        "  (none found: the spec needs an `Acceptance criteria` list before any test is written)"
    )
    return f"""Plan this feature against its spec. Do not write or change any code.

SPEC:
{state.issue}

Acceptance criteria found in the spec:
{listed}

Write the plan under exactly these headings:
1. Where it fits: the existing code this touches and the extension point
   that suits it best, with file paths.
2. Interface: names, arguments, defaults, and what stays backward compatible.
3. Edge cases and invalid input, and how each is handled.
4. Acceptance tests: for each criterion above, the test that will prove it.
   Say so if a criterion cannot be tested as written.
5. Smallest slice: the least that delivers working, tested behaviour, and
   what is deferred.
6. Open questions: anything the spec does not answer. Do not guess; ask.

Base every statement on the code you read. Keep it short enough to read in
two minutes."""


def _acceptance_prompt(state: FlowState) -> str:
    criteria = acceptance_criteria(state.issue)
    listed = "\n".join(f"  {n}. {c}" for n, c in enumerate(criteria, 1))
    plan = f"\n\nPLAN (already agreed):\n{state.plan}" if state.plan else ""
    return f"""Write acceptance tests for this feature. Do not implement it.

SPEC:
{state.issue}{plan}

Acceptance criteria to cover:
{listed}

- One test per criterion where practical; name each test after its criterion.
- Follow the existing test conventions and fixtures.
- Call the feature through the interface the spec describes, even though it
  does not exist yet. The tests are expected to fail for that reason.
- Do not add the feature, stubs or placeholders to make anything pass.

Tell me one test id that runs all of them (a test file, or a single test).
I will run it myself with `/flow repro <test id>` and it must fail because
the feature is missing."""


def _build_prompt(state: FlowState) -> str:
    red = state.repro_red or {}
    plan = f"\n\nPLAN (already agreed):\n{state.plan}" if state.plan else ""
    return f"""Build this feature: the smallest change that satisfies its spec.

SPEC:
{state.issue}{plan}

Failing acceptance tests: {state.repro_target}
Current result: {red.get("summary", "fails")}

- Build what the acceptance criteria ask for, and nothing they do not.
- Keep existing public APIs and unrelated behaviour unchanged.
- No unrelated refactoring and no new dependencies unless the spec names them.
- Do not weaken or delete any test. If an acceptance test is wrong, stop and
  tell me which and why.

Do not claim the tests pass. I will run `/flow verify` myself."""


def _repro_prompt(state: FlowState) -> str:
    if state.kind == "feature":
        return _acceptance_prompt(state)
    return f"""Write the smallest regression test that demonstrates this bug.

ISSUE:
{state.issue}

- Follow the existing test conventions and fixtures.
- Exercise the real code path; do not mock the behaviour under test.
- Cover the boundary case the issue describes.
- Do not change the implementation.

Tell me the exact test id to run. I will run it myself with
`/flow repro <test id>` and it must fail for the expected reason."""


def _fix_prompt(state: FlowState) -> str:
    if state.kind == "feature":
        return _build_prompt(state)
    red = state.repro_red or {}
    return f"""Implement the smallest correct fix for this issue.

ISSUE:
{state.issue}

Failing regression test: {state.repro_target}
Current result: {red.get("summary", "fails")}

- Fix the root cause, not just what this one test checks.
- Keep public APIs and unrelated behaviour unchanged.
- No unrelated refactoring and no new dependencies.
- Do not weaken or delete any existing test. If one must change, stop and
  tell me which and why.

Do not claim the tests pass. I will run `/flow verify` myself."""


def _review_fix_prompt(state: FlowState, direction: str) -> str:
    scope = (
        f"The developer's direction on which findings to act on:\n{direction}"
        if direction else
        "Act only on findings that are real defects in this change. For each "
        "finding you do not act on, say why in one line."
    )
    return f"""A reviewer examined the fix for this issue. Address the review.

ISSUE:
{state.issue}

REVIEW FINDINGS:
{state.review_findings}

{scope}

- Check each finding against the code before changing anything; a reviewer
  can be wrong.
- Keep changes minimal and inside the scope of this issue. Out-of-scope
  findings are noted for the PR description, not fixed here.
- Add or extend a test for any behaviour you change.
- Do not weaken or delete an existing test.

Do not claim the tests pass. I will run `/flow verify` and `/flow review`
again myself."""


def _refix_prompt(state: FlowState, failure: str) -> str:
    return f"""{_fix_prompt(state)}

A PREVIOUS FIX ATTEMPT IS ALREADY IN THE WORKING TREE AND DID NOT PASS:
{failure}

Before editing anything, explain why the previous change was not enough: is
there a second cause, or was the diagnosis wrong? Look at where the actual and
expected values first diverge. Then make the smallest additional change. Keep
the earlier change only if it is still correct."""


def _feature_review_prompt(state: FlowState, diff: str) -> str:
    return f"""Review this diff as a skeptical senior engineer checking a new
feature against its spec. Read the surrounding code where you need to; do not
modify files.

SPEC:
{state.issue}

Check:
1. Each acceptance criterion in the spec: is it implemented, and is there a
   test that would fail without it? Go through them one by one.
2. Behaviour the diff adds that the spec does not ask for.
3. Invalid input and boundary cases the tests do not cover.
4. Regressions in existing behaviour and public interfaces.
5. Tests that pass without proving the intended behaviour.
6. Anything in the diff the feature does not need.

List concrete findings by severity with file and line. If there is nothing
significant, say so rather than inventing problems.

DIFF:
{diff}"""


def _review_prompt(state: FlowState, diff: str) -> str:
    if state.kind == "feature":
        return _feature_review_prompt(state, diff)
    return f"""Review this diff as a skeptical senior engineer who assumes it
contains a subtle bug. Read the surrounding code where you need to; do not
modify files.

ORIGINAL ISSUE:
{state.issue}

Check:
1. Does the change resolve every symptom the issue lists, or only the one
   that was tested?
2. Boundary conditions the new test does not cover.
3. Other callers or code paths with the same flaw.
4. Regressions in adjacent behaviour.
5. Tests, new or existing, that pass without proving the intended behaviour.
6. Anything in the diff that the fix does not need.

List concrete findings by severity with file and line. If there is nothing
significant, say so rather than inventing problems.

DIFF:
{diff}"""


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def _onboard(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    stale = state.baseline or {}
    if stale.get("returncode") and stale.get("passed") is None and stale.get("failed") is None:
        # A baseline saved by an earlier run where the tests never started:
        # forget it and the command that produced it.
        state.baseline = None
        state.test_command = ""
    if state.baseline is not None and state.issue and not state.pr_url and not force:
        # The baseline is "the suite before this issue's changes". Recording
        # it again part-way through would count the new regression test (and
        # the fix) as the starting point, and verify would then report a
        # test missing. Seen live: baseline went 13 -> 14 and a good fix
        # failed verify.
        command_next, where = next_step(state, root)
        return FlowResult(
            f"**Onboard skipped:** already onboarded, and you are part-way "
            f"through \"{state.issue_title}\". The baseline stays at "
            f"{state.baseline.get('summary')}, recorded before this issue's "
            f"changes.\n\n{where}\n\n**Do next:** `{command_next}`\n\n"
            "To re-record the baseline anyway: `/flow onboard --force`",
            state, ok=False,
        )
    command = args.strip() or state.test_command or detect_test_command(root)
    _, branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    _, dirty = _git(root, "status", "--porcelain")
    found = [name for name in _INSTRUCTION_FILES if (root / name).exists()]

    state.test_command = command
    state.base_branch = state.base_branch or branch
    lines = [
        "**Onboard**",
        f"- Branch: `{branch or 'not a git repo'}`"
        + (" (uncommitted changes present)" if dirty else ""),
        f"- Instruction files: {', '.join(found) or 'none found'}",
    ]
    if not command:
        save_state(root, state)
        lines.append(
            "- Test command: **not detected**. Run `/flow onboard <command>` "
            "to set it, e.g. `/flow onboard make test`."
        )
        return FlowResult("\n".join(lines), state, ok=False)

    run, output = run_tests(root, command)
    if run.returncode != 0 and run.passed is None and run.bad is None:
        # Nothing ran at all (missing dependency, bad command, collection
        # crash). That is an environment problem, not a baseline: recording
        # it would let every later phase "match" a suite that never started.
        if not args.strip():
            state.test_command = ""  # re-detect next time
        save_state(root, state)
        lines += [
            f"- Test command: `{run.command}`",
            f"- **The tests could not start** (exit {run.returncode}). This is an "
            "environment problem, not a test failure, so no baseline was recorded.",
            _tail(output, 15),
            "Fix the environment (install the project's dependencies), or name "
            "the right command: `/flow onboard <command>`.",
        ]
        return FlowResult("\n".join(lines), state, ok=False)
    state.baseline = asdict(run)
    save_state(root, state)
    lines += [
        f"- Test command: `{run.command}`",
        f"- Baseline: {run.summary} (exit {run.returncode}, {run.seconds}s)",
    ]
    if run.returncode != 0:
        lines.append(
            "- The suite is **not green before any change**. Later phases "
            "compare against this baseline instead of requiring zero failures."
        )
        lines.append(_tail(output, 15))
    return FlowResult("\n".join(lines), state, agent_prompt=_recon_prompt(state, found))


def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "issue"


_ISSUE_DIRS = ("issues", "ISSUES", ".issues", "docs/issues", "bugs", "tickets")
_ISSUE_SUFFIXES = (".md", ".markdown", ".txt", ".rst")


def _first_line_title(text: str, fallback: str) -> str:
    return next(
        (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), fallback
    )


def find_issues(root: Path, *, remote: bool = True) -> list[tuple[str, str]]:
    """Issues the user could pick, as (reference, title) pairs.

    A reference is whatever `/flow issue <reference>` accepts: a path
    relative to ROOT for issue files checked into the repo, or `#N` for
    open GitHub issues (only when `gh` works here; skipped silently if not).
    """
    found: list[tuple[str, str]] = []
    seen: set[tuple[int, int]] = set()
    for name in _ISSUE_DIRS:
        folder = root / name
        if not folder.is_dir():
            continue
        # On a case-insensitive filesystem `issues` and `ISSUES` are the
        # same directory; identify it by inode so it is listed once.
        info = folder.stat()
        if (info.st_dev, info.st_ino) in seen:
            continue
        seen.add((info.st_dev, info.st_ino))
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _ISSUE_SUFFIXES:
                continue
            if path.stem.lower() in ("readme", "template", "index"):
                continue
            try:
                title = _first_line_title(path.read_text(errors="replace"), path.stem)
            except OSError:
                title = path.stem
            found.append((str(path.relative_to(root)), title[:100]))
    if remote and shutil.which("gh"):
        code, out = _run(
            ["gh", "issue", "list", "--state", "open", "--limit", "20",
             "--json", "number,title"],
            root, timeout=10.0,
        )
        if code == 0:
            try:
                for item in json.loads(out):
                    found.append((f"#{item['number']}", str(item["title"])[:100]))
            except (ValueError, KeyError, TypeError):
                pass
    return found


def _match_local_issue(root: Path, arg: str) -> Optional[str]:
    """Resolve a short ARG ("2", "002", "decode") to one issue file.

    Numbers match the file's leading number; words match a unique substring
    of the file name. Returns None unless exactly one file matches.
    """
    if not arg or len(arg) > 60 or any(c.isspace() for c in arg):
        return None
    files = [ref for ref, _ in find_issues(root, remote=False)]
    key = arg.lstrip("#")
    if key.isdigit():
        hits = []
        for ref in files:
            lead = re.match(r"\d+", Path(ref).name)
            if lead and int(lead.group()) == int(key):
                hits.append(ref)
    else:
        hits = [ref for ref in files if key.lower() in Path(ref).name.lower()]
    return hits[0] if len(hits) == 1 else None


def _issue_menu(root: Path) -> str:
    issues = find_issues(root)
    if not issues:
        linear_hint = (
            " Or pick one from Linear: `/flow linear`." if _linear().configured() else ""
        )
        return (
            "No issue files or open GitHub issues found here. Give the issue "
            "directly: `/flow issue <text | path/to/issue.md | #123 | issue URL>`"
            + linear_hint
        )
    lines = ["**Pick an issue:**", ""]
    for ref, title in issues:
        lines.append(f"- `/flow issue {ref}` {title}")
    lines += ["", "Or paste the issue text: `/flow issue <text>`"]
    if _linear().configured():
        lines.append("From Linear: `/flow linear` (yours) or `/flow linear <project>`")
    return "\n".join(lines)


def _linear():
    """The Linear module, imported on first use so `/flow` stays importable
    (and fast) without the MCP client."""
    from bach.ops import linear

    return linear


def _linear_menu(project: str) -> str:
    linear = _linear()
    issues, mine = linear.list_issues(project)
    scope = f"open issues in **{project}**" if project else "open issues assigned to you"
    if not issues:
        return f"Linear has no {scope}." + _linear_projects(project)
    lines = [f"**Linear: {scope}** ({len(issues)})", ""]
    for issue in issues:
        detail = [issue.status]
        if not project and issue.project:
            detail.append(issue.project)
        if project:
            detail.append("you" if issue.identifier in mine else (issue.assignee or "unassigned"))
        lines.append(
            f"- `/flow issue {issue.identifier}` {issue.title} "
            f"({', '.join(d for d in detail if d)})"
        )
    lines += ["", "Picking one starts the workflow and moves it to In Progress in Linear."]
    return "\n".join(lines) + _linear_projects(project)


def _linear_projects(project: str) -> str:
    """With no project chosen, the projects to choose from."""
    if project:
        return ""
    try:
        names = _linear().list_projects()
    except Exception:  # noqa: BLE001 - the issue list above is still useful
        return ""
    if not names:
        return ""
    return "\n\n**Or list a whole project:**\n\n" + "\n".join(f"- `/flow linear {n}`" for n in names)


def _resolve_issue(root: Path, arg: str) -> tuple[str, str]:
    """ARG is a file path, a short match for an issue file, a GitHub issue
    (#12, 12, or URL), or literal text.

    Returns (title, full text).
    """
    local = _match_local_issue(root, arg)
    if local and not (root / arg).is_file():
        arg = local
    candidate = root / arg
    if len(arg) < 300 and "\n" not in arg and candidate.is_file():
        text = candidate.read_text()
        title = next(
            (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()),
            candidate.stem,
        )
        return title, text
    if re.fullmatch(r"#?\d+", arg) or re.match(r"https://github\.com/.+/issues/\d+", arg):
        code, out = _run(
            ["gh", "issue", "view", arg.lstrip("#"), "--json", "title,body"], root
        )
        if code == 0:
            data = json.loads(out)
            return data["title"], f"{data['title']}\n\n{data.get('body') or ''}"
        raise ValueError(f"could not read issue {arg}: {out.strip()[:200]}")
    return arg.splitlines()[0][:80], arg


def _issue(
    root: Path, state: FlowState, args: str, force: bool, provider: str = "", *, kind: str = "bug"
) -> FlowResult:
    command = "feature" if kind == "feature" else "issue"
    if not args.strip():
        menu = _issue_menu(root)
        if kind == "feature":
            menu = menu.replace("/flow issue", "/flow feature").replace(
                "**Pick an issue:**", "**Pick the feature's issue:**")
        return FlowResult(menu, state, ok=False)
    if state.baseline is None and not force:
        return _blocked(
            state, "issue", "no test baseline recorded.",
            "Run `/flow onboard` first so later phases can tell a regression "
            "from a pre-existing failure.",
        )
    arg = args.strip()
    linear_issue = None
    linear = _linear()
    ref = linear.parse_reference(arg)
    if ref and linear.configured() and not (root / arg).is_file() \
            and not _match_local_issue(root, arg):
        try:
            linear_issue = linear.get_issue(ref)
        except linear.LinearError as exc:
            return FlowResult(f"**/flow issue failed:** {exc}", state, ok=False)
        title = f"{linear_issue.identifier}: {linear_issue.title}"
        text = "\n\n".join(filter(None, [
            title, linear_issue.description.strip(),
            f"Linear: {linear_issue.url}" if linear_issue.url else "",
        ]))
    else:
        try:
            title, text = _resolve_issue(root, arg)
        except ValueError as exc:
            return FlowResult(f"**/flow issue failed:** {exc}", state, ok=False)

    # Guard against restarting by accident. Re-running the same issue, or
    # switching issues mid-way, would wipe the recorded progress and send
    # the model off to investigate and write tests a second time.
    in_progress = bool(state.issue) and not state.pr_url and not state.pr_drafted and (
        state.repro_requested or state.repro_red is not None or state.fix_requested
    )
    same = state.issue.strip() == text.strip() or (
        linear_issue is not None and state.issue_ref == linear_issue.identifier
    )
    if state.issue and same and not force:
        command, where = next_step(state, root)
        return FlowResult(
            f"**Already working on this issue:** {state.issue_title}\n\n"
            f"{where}\n\n**Do next:** `{command}`\n\n"
            "Nothing was changed. To throw away the progress and start this "
            f"issue again: `/flow {command} {args.strip()} --force`",
            state, ok=False,
        )
    if in_progress and not force:
        return FlowResult(
            f"**/flow issue is blocked:** you are part-way through "
            f"\"{state.issue_title}\" (currently in {state.current()}).\n\n"
            "Finish it first, or abandon it and switch with "
            f"`/flow {command} {args.strip()} --force`. Abandoning does not revert "
            "files: check `git status` for changes left behind.",
            state, ok=False,
        )

    # A new issue restarts everything downstream of onboarding.
    fresh = FlowState(
        test_command=state.test_command,
        base_branch=state.base_branch,
        baseline=state.baseline,
        issue=text,
        issue_title=title,
        issue_ref=linear_issue.identifier if linear_issue else "",
        issue_url=linear_issue.url if linear_issue else "",
        kind=kind,
    )
    lines = [f"**{'Feature' if kind == 'feature' else 'Issue'}:** {title}"]
    _, current = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if current and current == fresh.base_branch:
        # Linear suggests a branch name that its GitHub integration links
        # back to the issue automatically; use it when there is one.
        branch = (linear_issue.branch if linear_issue else "") or (
            f"{'feat' if kind == 'feature' else 'fix'}/{_slug(title)}")
        code, out = _git(root, "checkout", "-b", branch)
        if code == 0:
            fresh.branch = branch
            lines.append(f"- Created branch `{branch}` from `{current}`")
        else:
            lines.append(f"- Could not create branch `{branch}`: {out}")
    elif current:
        fresh.branch = current
        lines.append(f"- Staying on existing branch `{current}`")
    save_state(root, fresh)
    if linear_issue is not None:
        try:
            lines.append(f"- {linear.start_issue(linear_issue)}")
        except linear.LinearError as exc:
            # The workflow does not depend on the tracker update; say so and go on.
            lines.append(f"- Could not update {linear_issue.identifier} in Linear: {exc}")
    if kind == "feature":
        criteria = acceptance_criteria(text)
        if criteria:
            lines.append(f"- Acceptance criteria in the spec: {len(criteria)}")
        else:
            lines.append(
                "- **No acceptance criteria in the spec.** Add an `Acceptance "
                "criteria` list to the issue; `/flow repro` will not write "
                "tests without one."
            )
        lines.append("- Planning against the spec (no code)")
        return FlowResult("\n".join(lines), fresh, agent_prompt=_plan_prompt(fresh))
    lines.append("- Investigating root cause (read-only)")
    return FlowResult("\n".join(lines), fresh, agent_prompt=_investigate_prompt(fresh))


def _feature(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    """`/flow feature <ENG-123 | file | text>`: step 2 for a feature. The
    issue is the spec; the model plans against it instead of tracing a bug."""
    return _issue(root, state, args, force, provider, kind="feature")


def record_plan(project_path: str | Path, plan: str) -> tuple[bool, str]:
    """Keep the plan the model wrote for a feature, and post it to the
    tracker issue as a comment so the design is on record before any code.

    Returns (ok, message for the person at the keyboard).
    """
    root = project_root(project_path)
    state = load_state(root)
    plan = plan.strip()
    if state.kind != "feature" or not state.issue:
        return False, "**Plan not recorded:** no feature is in progress. Start one with `/flow feature <issue>`."
    if not plan:
        return False, "**Plan not recorded:** the plan was empty."
    state.plan = plan
    state.plan_posted = False
    lines = ["**Plan recorded.**"]
    linear = _linear()
    if state.issue_ref and linear.configured():
        try:
            linear.add_comment(state.issue_ref, f"**Plan** (from `/flow feature`)\n\n{plan}")
            state.plan_posted = True
            lines.append(f"- Posted to {state.issue_ref} in Linear as a comment")
        except linear.LinearError as exc:
            lines.append(f"- Could not post to {state.issue_ref} in Linear: {exc}")
    elif state.issue_ref:
        lines.append(f"- Not posted to {state.issue_ref}: Linear is not set up (LINEAR_API_KEY)")
    save_state(root, state)
    command, where = next_step(state, root)
    lines += ["", where, "", f"**Do next:** `{command}`"]
    return True, "\n".join(lines)


def _refresh_spec(root: Path, state: FlowState) -> str:
    """Re-read a feature's spec from Linear, so criteria added to the issue
    after it was picked are seen. Returns a note for the output ("" if none)."""
    linear = _linear()
    if not (state.issue_ref and linear.configured()):
        return ""
    try:
        issue = linear.get_issue(state.issue_ref)
    except linear.LinearError as exc:
        return f"- Could not re-read {state.issue_ref} from Linear ({exc}); using the saved copy"
    state.issue_title = f"{issue.identifier}: {issue.title}"
    state.issue = "\n\n".join(filter(None, [
        state.issue_title, issue.description.strip(),
        f"Linear: {issue.url}" if issue.url else "",
    ]))
    save_state(root, state)
    return f"- Spec re-read from {state.issue_ref} in Linear"


def _linear_phase(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    """`/flow linear [project]`: list Linear issues to pick from. Selecting
    is `/flow issue <ID>`, the same path as every other issue source."""
    linear = _linear()
    if not linear.configured():
        return FlowResult(
            "**Linear is not set up.** Export `LINEAR_API_KEY` (a personal API "
            "key from Linear: Settings, Security & access) in your shell "
            "profile, then restart the editor or terminal Bach runs in.",
            state, ok=False,
        )
    try:
        return FlowResult(_linear_menu(args.strip()), state)
    except linear.LinearError as exc:
        return FlowResult(f"**/flow linear failed:** {exc}", state, ok=False)


def _repro(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    if not state.issue and not force:
        return _blocked(state, "repro", "no issue captured.", "Run `/flow issue <...>` first.")
    target = args.strip()
    if not target and not force and state.repro_red is not None:
        return FlowResult(
            f"**A regression test is already recorded:** `{state.repro_target}` "
            f"({state.repro_red.get('summary')}).\n\nAsking again would add a "
            "second, duplicate test. **Do next:** "
            f"`{next_step(state, root)[0]}`\n\nTo replace the test instead: "
            "`/flow repro --force`",
            state, ok=False,
        )
    if not target and not force and state.repro_requested:
        return FlowResult(
            "**A regression test was already requested** and has not been run "
            "yet. Asking again would add a duplicate.\n\n**Do next:** "
            "`/flow repro <test id>` with the id from the model's reply (look "
            "for the new test in `git status` / `git diff` if you lost it).\n\n"
            "To have it rewritten instead: `/flow repro --force`",
            state, ok=False,
        )
    feature = state.kind == "feature"
    spec_note = ""
    if feature and not target:
        spec_note = _refresh_spec(root, state)
        criteria = acceptance_criteria(state.issue)
        if not criteria and not force:
            return FlowResult(
                "**/flow repro is blocked:** the spec has no acceptance criteria.\n\n"
                + (spec_note + "\n\n" if spec_note else "")
                + "Acceptance tests are written from the criteria, so there is "
                "nothing to test yet. Add a section to "
                + (f"{state.issue_ref} in Linear" if state.issue_ref else "the issue")
                + " like this, then run `/flow repro` again:\n\n"
                "```\n## Acceptance criteria\n"
                "- <something a test can check>\n- <another>\n```",
                state, ok=False,
            )
    if not target:
        replacing = state.repro_requested or state.repro_red is not None
        state.repro_requested = True
        state.repro_red = None
        save_state(root, state)
        if replacing:
            return FlowResult(
                "**Repro (replacing):** asking for the regression test to be "
                "rewritten in place. Run `/flow repro <test id>` afterwards.",
                state,
                agent_prompt=_repro_prompt(state) + (
                    "\n\nA regression test for this issue was already added to "
                    "the working tree"
                    + (f" (`{state.repro_target}`)" if state.repro_target else "")
                    + ". Replace or correct that test. Do not leave two tests "
                    "for the same bug."
                ),
            )
        if feature:
            count = len(acceptance_criteria(state.issue))
            return FlowResult(
                f"**Acceptance tests:** asking for tests from the {count} "
                "acceptance criteria in the spec."
                + (f"\n{spec_note}" if spec_note else "")
                + "\nWhen they are written, run `/flow repro <test id>` to "
                "confirm they fail.",
                state, agent_prompt=_repro_prompt(state),
            )
        return FlowResult(
            "**Repro:** asking for a regression test. When it is written, run "
            "`/flow repro <test id>` to confirm it is red.",
            state, agent_prompt=_repro_prompt(state),
        )
    if not state.test_command:
        return _blocked(state, "repro", "no test command.", "Run `/flow onboard` first.")

    run, output = run_tests(root, state.test_command, target)
    if run.returncode == 0:
        return FlowResult(
            f"**Repro rejected:** `{run.command}` passed ({run.summary}).\n\n"
            "A regression test has to fail before the fix exists. Either it "
            "does not exercise the bug or the bug is not what we think it is.",
            state, ok=False,
        )
    if run.bad == 0 or (run.bad is None and run.returncode in (4, 5, 124, 127)):
        return FlowResult(
            f"**Repro rejected:** `{run.command}` did not run any failing test "
            f"(exit {run.returncode}: {run.summary}). Check the test id.\n\n{_tail(output, 10)}",
            state, ok=False,
        )
    state.repro_target = target
    state.repro_red = asdict(run)
    save_state(root, state)
    if feature:
        return FlowResult(
            f"**Acceptance tests are red:** `{run.command}` -> {run.summary}\n\n{_tail(output)}\n\n"
            "Read the failure above. It should fail because the feature does "
            "not exist yet (a missing function or command counts). If it fails "
            "for a mistake in the test itself, fix the test and re-run "
            "`/flow repro`. Otherwise continue with `/flow fix`.",
            state,
        )
    return FlowResult(
        f"**Repro is red:** `{run.command}` -> {run.summary}\n\n{_tail(output)}\n\n"
        "Read the failure above. If it fails for the reason the issue "
        "describes, continue with `/flow fix`. If it fails for another reason "
        "(import error, bad fixture), fix the test and re-run `/flow repro`.",
        state,
    )


def _fix(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    if state.repro_red is None and not force:
        return _blocked(
            state, "fix", "there is no failing regression test.",
            "Run `/flow repro`, then `/flow repro <test id>` to record a red test.",
        )
    previous_failure = state.verify_failed
    findings = (
        state.review_findings
        if state.reviewed_diff and not state.review_addressed and not previous_failure
        else ""
    )
    state.pr_drafted = False
    if findings:
        state.review_addressed = True
        state.author = provider or state.author
        state.verified = None
        state.verified_diff = ""
        state.reviewed_diff = ""
        save_state(root, state)
        return FlowResult(
            "**Fix (review findings):** handing the reviewer's findings to the "
            "model" + (f", with your direction: {args.strip()}" if args.strip() else "")
            + ". Read what it changes, then `/flow verify` and `/flow review` again.",
            state, agent_prompt=_review_fix_prompt(state, args.strip()),
        )
    state.fix_requested = True
    state.author = provider
    state.verified = None
    state.verify_failed = ""
    state.verified_diff = ""
    state.reviewed_diff = ""
    save_state(root, state)
    if previous_failure:
        return FlowResult(
            "**Fix (another attempt):** the last verify failed, so the model is "
            "given that failure and asked why the first change was not enough. "
            "Run `/flow verify` when it is applied.",
            state, agent_prompt=_refix_prompt(state, previous_failure),
        )
    if state.kind == "feature":
        return FlowResult(
            "**Build:** requesting the smallest change that satisfies the "
            "spec. Run `/flow verify` when it is applied.",
            state, agent_prompt=_fix_prompt(state),
        )
    return FlowResult(
        "**Fix:** requesting the smallest correct change. Run `/flow verify` "
        "when it is applied.",
        state, agent_prompt=_fix_prompt(state),
    )


def _verify(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    if not state.test_command:
        return _blocked(state, "verify", "no test command.", "Run `/flow onboard` first.")
    lines = ["**Verify**"]
    problems: list[str] = []

    if state.repro_target:
        repro, output = run_tests(root, state.test_command, state.repro_target)
        ok = repro.returncode == 0
        lines.append(f"- Regression test `{state.repro_target}`: {'green' if ok else 'STILL RED'} ({repro.summary})")
        if not ok:
            problems.append("the regression test still fails")
            lines.append(_tail(output, 15))
    elif not force:
        return _blocked(
            state, "verify", "no regression test recorded.",
            "Without a test that went red -> green there is nothing proving the fix.",
        )

    # The rule is deliberately simple: the regression test passes, and
    # nothing fails now that was not already failing at the baseline.
    # Tests are compared by name, never by count, so adding tests or
    # re-recording the baseline cannot produce a false failure.
    full, output = run_tests(root, state.test_command)
    base = _suite(state.baseline)
    lines.append(f"- Full suite: {full.summary} (exit {full.returncode}, {full.seconds}s)")
    known = set(base.failing) if base else set()
    newly_failing = [t for t in full.failing if t not in known]
    if newly_failing:
        shown = ", ".join(newly_failing[:6]) + (" ..." if len(newly_failing) > 6 else "")
        problems.append(f"{len(newly_failing)} test(s) fail that did not fail before: {shown}")
    elif full.returncode != 0 and not full.failing:
        # The runner did not name its failures; fall back to what it did say.
        if base is None or base.returncode == 0:
            problems.append("the test command fails and passed at the baseline")
        elif full.bad is not None and base.bad is not None and full.bad > base.bad:
            problems.append(f"{full.bad - base.bad} more failing test(s) than the baseline")
    if base is not None and base.failing:
        lines.append(
            f"- Already failing before this change (not counted): {', '.join(base.failing[:6])}"
        )
    deleted = _deleted_test_files(root, _base_ref(root, state))
    if deleted:
        problems.append(f"existing test file(s) were deleted: {', '.join(deleted[:6])}")
    removed = [r for r in _removed_test_lines(root, _base_ref(root, state)) if r[0] not in deleted]
    if removed:
        lines.append(
            "- **Check:** lines were removed from existing tests ("
            + ", ".join(f"{path}: -{count}" for path, count in removed[:6])
            + "). Make sure no test was deleted or weakened to get here."
        )

    if problems:
        state.verified = None
        state.verified_diff = ""
        state.verify_failed = "; ".join(problems)
        save_state(root, state)
        lines.append(_tail(output, 20))
        lines.append("**Not verified:** " + "; ".join(problems) + ".")
        lines.append(
            "The fix in the working tree is not enough. Read the failure above "
            "(where do actual and expected first differ?) and decide whether "
            "there is a second cause or the diagnosis was wrong. Then "
            "`/flow fix` asks for another attempt and passes this failure along."
        )
        return FlowResult("\n".join(lines), state, ok=False)

    state.verify_failed = ""
    state.verified = asdict(full)
    state.verified_diff = diff_fingerprint(root, _base_ref(root, state))
    save_state(root, state)
    # Not always review: an earlier review of this exact diff may still stand.
    command, where = next_step(state, root)
    lines.append(
        "**Verified:** the acceptance tests pass and nothing else broke."
        if state.kind == "feature" else
        "**Verified:** the regression test passes and nothing else broke."
    )
    if command != "/flow review":
        lines.append(where)
    lines.append(f"**Do next:** `{command}`")
    return FlowResult("\n".join(lines), state)


def _suite(raw: Optional[dict]) -> Optional[SuiteRun]:
    """A saved SuiteRun, tolerating state written by older versions."""
    if not raw:
        return None
    known = {k: v for k, v in raw.items() if k in SuiteRun.__dataclass_fields__}
    return SuiteRun(**known)


def _is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    in_test_dir = any(
        seg in ("test", "tests", "__tests__", "spec") for seg in path.lower().split("/")[:-1]
    )
    return (
        in_test_dir or name.startswith("test_") or "_test." in name
        or ".test." in name or ".spec." in name
    )


def _deleted_test_files(root: Path, base: str) -> list[str]:
    """Test files that existed at BASE and are gone now."""
    code, out = _git(root, "diff", base, "--relative", "--name-status")
    if code != 0:
        return []
    return [
        line.split("\t", 1)[1] for line in out.splitlines()
        if line.startswith("D\t") and _is_test_path(line.split("\t", 1)[1])
    ]


def _removed_test_lines(root: Path, base: str) -> list[tuple[str, int]]:
    """(path, lines removed) for existing test files this change cut lines
    from. A direct look at "was a test deleted or weakened", instead of
    inferring it from test counts."""
    code, out = _git(root, "diff", base, "--relative", "--numstat")
    if code != 0:
        return []
    found = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[1].isdigit() or int(parts[1]) == 0:
            continue
        if _is_test_path(parts[2]):
            found.append((parts[2], int(parts[1])))
    return found


def _failing_ids(output: str) -> list[str]:
    """Test ids from pytest's `FAILED path::name` / `ERROR path::name` lines."""
    ids = []
    for match in re.finditer(r"^(?:FAILED|ERROR) (\S+)", output, re.M):
        if match.group(1) not in ids:
            ids.append(match.group(1))
    return ids


def _stale(root: Path, state: FlowState) -> bool:
    """True when the code differs from what `/flow verify` last tested."""
    if not state.verified_diff:
        return False
    return state.verified_diff != diff_fingerprint(root, _base_ref(root, state))


def _review(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    if state.verified is None and not force:
        return _blocked(state, "review", "the change has not been verified.", "Run `/flow verify` first.")
    if _stale(root, state) and not force:
        return _blocked(
            state, "review", "the code changed after `/flow verify`.",
            "Re-run `/flow verify` so the review covers code that was actually tested.",
        )
    tokens = args.split()
    if "--with" in tokens:
        index = tokens.index("--with")
        if index + 1 >= len(tokens):
            return FlowResult("Usage: `/flow review [--with <provider>]`", state, ok=False)
        state.reviewer = tokens[index + 1]
    base = _base_ref(root, state)
    diff = working_diff(root, base)
    if not diff:
        return FlowResult("**Review:** there is no diff to review.", state, ok=False)
    if len(diff) > _MAX_DIFF_CHARS:
        _, stat = _git(root, "diff", base, "--stat")
        diff = (
            diff[:_MAX_DIFF_CHARS]
            + f"\n\n[diff truncated; full stat follows]\n{stat}"
        )
    # The phase is not marked done here. It is done when a reviewer has
    # actually returned findings and the surface calls `record_review`.
    state.reviewed_diff = ""
    state.reviewed_by = ""
    save_state(root, state)
    return FlowResult(
        "**Review:** handing the diff and the original issue to a separate "
        "read-only reviewer. If it finds something real, fix it and go back "
        "through `/flow verify`.",
        state,
        review_prompt=_review_prompt(state, diff),
        reviewer=state.reviewer,
        review_fingerprint=diff_fingerprint(root, base),
    )


def record_review(
    project_path: str | Path, fingerprint: str, reviewed_by: str, findings: str = ""
) -> FlowState:
    """Mark the review phase done for the code identified by FINGERPRINT.

    Called by a surface after the reviewer has returned its FINDINGS, which
    are kept so a later `/flow fix` can address them.
    """
    root = project_root(project_path)
    state = load_state(root)
    state.reviewed_diff = fingerprint
    state.reviewed_by = reviewed_by
    state.review_findings = findings.strip()[:12000]
    state.review_addressed = False
    save_state(root, state)
    return state


def _fixes_line(state: FlowState) -> str:
    """`Fixes ENG-123` so the tracker links the PR and closes the issue on merge."""
    if not state.issue_ref:
        return ""
    return f"\nFixes {state.issue_ref}\n"


def changed_files(root: Path, base: str) -> list[tuple[str, int, int, bool]]:
    """(path, lines added, lines removed, is new) for every file the change
    touches, paths relative to ROOT. Binary files count as 0/0."""
    _, out = _git(root, "diff", base, "--numstat", "--relative")
    files = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added = int(parts[0]) if parts[0].isdigit() else 0
        removed = int(parts[1]) if parts[1].isdigit() else 0
        files.append((parts[2], added, removed, False))
    for name in _changed_paths(root, base)[1]:
        try:
            lines = len((root / name).read_text(errors="replace").splitlines())
        except OSError:
            lines = 0
        files.append((name, lines, 0, True))
    return sorted(files)


def _files_table(files: list[tuple[str, int, int, bool]]) -> str:
    if not files:
        return "_No files changed._"
    rows = ["| File | Added | Removed |", "| --- | ---: | ---: |"]
    for path, added, removed, new in files:
        rows.append(f"| `{path}`{' (new)' if new else ''} | +{added} | -{removed} |")
    total_added = sum(f[1] for f in files)
    total_removed = sum(f[2] for f in files)
    rows.append(f"| **{len(files)} file(s)** | **+{total_added}** | **-{total_removed}** |")
    return "\n".join(rows)


def _files_text(files: list[tuple[str, int, int, bool]]) -> str:
    """The same list for a terminal: aligned plain text, no table markup."""
    if not files:
        return "```\n(no files changed)\n```"
    names = [f"{path}{' (new)' if new else ''}" for path, _, _, new in files]
    total = f"{len(files)} file{'s' if len(files) != 1 else ''} changed"
    width = max(len(n) for n in names + [total])
    plus = max(len(str(v)) for v in [f[1] for f in files] + [sum(f[1] for f in files)])
    minus = max(len(str(v)) for v in [f[2] for f in files] + [sum(f[2] for f in files)])

    def row(name: str, added: int, removed: int) -> str:
        return f"{name:<{width}}   +{added:<{plus}}  -{removed:<{minus}}".rstrip()

    lines = [row(n, f[1], f[2]) for n, f in zip(names, files)]
    lines.append("-" * len(max(lines, key=len)))
    lines.append(row(total, sum(f[1] for f in files), sum(f[2] for f in files)))
    return "```\n" + "\n".join(lines) + "\n```"


def _files_summary(files: list[tuple[str, int, int, bool]]) -> str:
    return f"{len(files)} file(s), +{sum(f[1] for f in files)} -{sum(f[2] for f in files)}"


_PROSE = {
    "Problem": "_TODO: what was going wrong, in plain words._",
    "Root cause": "_TODO: one or two sentences._",
    "Fix": "_TODO: what changed and why it is the smallest correct change._",
    "Not verified": "_TODO: anything not tested, and follow-ups worth filing._",
}


def _written_sections(text: str) -> dict[str, str]:
    """The Root cause / Fix / Not verified sections of an earlier draft that
    have real content, so regenerating the draft does not throw away wording
    that was already written or corrected."""
    kept: dict[str, str] = {}
    for chunk in re.split(r"^## ", text, flags=re.M)[1:]:
        heading, _, body = chunk.partition("\n")
        body = body.strip()
        if heading.strip() in _PROSE and body and "_TODO:" not in body:
            kept[heading.strip()] = body.split("\n<details>")[0].strip()
    return kept


def pr_body(root: Path, state: FlowState, keep: Optional[dict[str, str]] = None) -> str:
    """The pull request description. Facts (problem, files, test results)
    come from the recorded state; KEEP supplies prose sections already
    written, and the rest are left as `_TODO:` for the model to write."""
    keep = keep or {}
    files = changed_files(root, _base_ref(root, state))
    base = state.baseline or {}
    red = state.repro_red or {}
    verified = state.verified or {}
    issue = state.issue.strip()
    if len(issue) > 1500:
        issue = issue[:1500] + "\n[...]"
    prose = {name: keep.get(name, todo) for name, todo in _PROSE.items()}
    return f"""## Problem

{prose["Problem"]}
{_fixes_line(state)}
## Root cause

{prose["Root cause"]}

## Fix

{prose["Fix"]}

## Files changed

{_files_table(files)}

## Verification

| Check | Result |
| --- | --- |
| Regression test `{state.repro_target or "n/a"}` before fix | {red.get("summary", "not recorded")} |
| Regression test after fix | passes |
| Full suite before (`{base.get("command", "n/a")}`) | {base.get("summary", "not recorded")} |
| Full suite after | {verified.get("summary", "not recorded")} |
| Fix written with | {state.author or "not recorded"} |
| Reviewed by | {state.reviewed_by or "not reviewed"} |

## Not verified

{prose["Not verified"]}

<details>
<summary>Issue as filed</summary>

{issue or "_No issue captured._"}

</details>
"""


def _pr_diff(root: Path, state: FlowState) -> FlowResult:
    """`/flow pr --diff`: the change itself, file by file."""
    base = _base_ref(root, state)
    files = changed_files(root, base)
    if not files:
        return FlowResult("**No changes** against " + f"`{state.base_branch or base}`.", state)
    lines = [f"**Changes** against `{state.base_branch or base}`", "", _files_text(files)]
    chunks = re.split(r"^(?=diff --git )", working_diff(root, base), flags=re.M)
    for chunk in chunks:
        if not chunk.strip():
            continue
        match = re.search(r"^\+\+\+ (?:b/)?(.+)$", chunk, flags=re.M)
        name = match.group(1).strip() if match else "(file)"
        body = chunk.rstrip().splitlines()
        if len(body) > 400:
            body = body[:400] + [f"[... {len(body) - 400} more lines; open the file for the rest ...]"]
        lines += ["", f"### `{name}`", "", "```diff", *body, "```"]
    return FlowResult("\n".join(lines), state)


def _ignore_locally(root: Path, name: str) -> None:
    """Have git ignore NAME in this clone only (`.git/info/exclude`), so the
    draft never shows up in `git status` or gets committed, and nothing has
    to be added to the repository's own .gitignore."""
    git_dir = _git_dir(root)
    if git_dir is None:
        return
    exclude = git_dir / "info" / "exclude"
    try:
        current = exclude.read_text() if exclude.exists() else ""
        if name not in current.split():
            exclude.parent.mkdir(parents=True, exist_ok=True)
            exclude.write_text(current + ("" if current.endswith("\n") or not current else "\n") + name + "\n")
    except OSError:
        pass


def _pr_prompt(root: Path, state: FlowState, draft: Path) -> str:
    diff = working_diff(root, _base_ref(root, state))
    if len(diff) > 12000:
        diff = diff[:12000] + "\n[... diff truncated; read the files for the rest ...]"
    findings = ""
    if state.review_findings and not state.review_addressed:
        findings = (
            "\nThe reviewer reported these findings and they were NOT acted on. "
            "Any that are real limitations belong under \"Not verified\":\n\n"
            f"{state.review_findings[:4000]}\n"
        )
    return f"""Finish the pull request description in `{draft.name}` ({draft}).

The file already has the Files changed table, the Verification table and
the issue as it was filed (at the bottom), filled in from recorded results.
Leave those exactly as they are, and leave any section that is already
written. Replace each line that starts with `_TODO:` with real content:

- **Problem**: two or three sentences in plain words. What the reporter saw
  and why it matters to them. Explain it; do not paste the issue.
- **Root cause**: one or two sentences. What was wrong in the code (name the
  file and function) and why that produced the reported symptom.
- **Fix**: what was changed and why it is the smallest correct change. Name
  the regression test that now guards it.
- **Not verified**: what was not tested or checked, stated plainly, and any
  follow-up worth filing. Do not write "nothing" unless that is true.

Base every statement on the diff below and the code. Do not claim any test
result that is not in the Verification table. Do not change any code or
tests: only `{draft.name}`.

When the file is written, print the Problem, Root cause, Fix and Not
verified sections exactly as they are in the file, and then stop. No preamble, no
summary of what you did, and nothing about the workflow tool or how this
command behaves.
{findings}
The change:

```diff
{diff}
```
"""


def _pr(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    create = "--create" in args.split()
    if "--diff" in args.split():
        return _pr_diff(root, state)
    if state.pr_url and not force:
        return FlowResult(
            f"**The pull request is already open:** {state.pr_url}\n\n"
            "Nothing was changed. `/flow pr --diff` shows what it contains. "
            "Pick the next issue with `/flow issue`.",
            state,
        )
    if not force:
        if state.verified is None:
            return _blocked(state, "pr", "the change has not been verified.", "Run `/flow verify`.")
        if not state.reviewed_diff:
            return _blocked(state, "pr", "the diff has not been reviewed.", "Run `/flow review`.")
        now = diff_fingerprint(root, _base_ref(root, state))
        if state.verified_diff != now or state.reviewed_diff != now:
            return _blocked(
                state, "pr", "the code changed after it was verified and reviewed.",
                "Re-run `/flow verify` and `/flow review` on the code you intend to ship.",
            )

    git_dir = _git_dir(root)
    if git_dir is None:
        return FlowResult("**/flow pr:** not a git repository.", state, ok=False)
    body_path = root / _PR_DRAFT
    shown = f"{scope_of(root)}/{_PR_DRAFT}" if scope_of(root) else _PR_DRAFT
    _ignore_locally(root, _PR_DRAFT)

    verb = "Add" if state.kind == "feature" else "Fix"
    title = f"{verb}: {state.issue_title}" if state.issue_title else verb
    branch = state.branch or _git(root, "rev-parse", "--abbrev-ref", "HEAD")[1]
    if not create:
        keep = _written_sections(body_path.read_text()) if body_path.exists() else {}
        body_path.write_text(pr_body(root, state, keep))
        state.pr_drafted = True
        save_state(root, state)
        files = changed_files(root, _base_ref(root, state))
        unwritten = [name for name in _PROSE if name not in keep]
        header = (
            f"**Pull request draft:** `{shown}`\n\n"
            f"- Title: `{title}`\n- Branch: `{branch}` -> `{state.base_branch}`\n"
            f"\n{_files_text(files)}\n\n"
            "Nothing has been committed or pushed. `/flow pr --diff` shows the "
            "changes line by line; `/flow pr --create` opens the pull request."
        )
        if not unwritten:
            # Already written on an earlier run: show it, do not just say so.
            base, red, verified = state.baseline or {}, state.repro_red or {}, state.verified or {}
            checks = [
                f"- Regression test before the fix: {red.get('summary', 'not recorded')}",
                "- Regression test after the fix: passes",
                f"- Full suite before: {base.get('summary', 'not recorded')}",
                f"- Full suite after: {verified.get('summary', 'not recorded')}",
                f"- Reviewed by: {state.reviewed_by or 'not reviewed'}",
            ]
            view = [
                "**Problem**", "", keep["Problem"], "",
                "**Root cause**", "", keep["Root cause"], "",
                "**Fix**", "", keep["Fix"], "",
                "**Verification**", "", *checks, "",
                "**Not verified**", "", keep["Not verified"],
            ]
            return FlowResult(header + "\n\n" + "\n".join(view), state)
        return FlowResult(header, state, agent_prompt=_pr_prompt(root, state, body_path))

    if not body_path.exists():
        return _blocked(
            state, "pr", "there is no pull request description yet.",
            "Run `/flow pr` first to draft it.",
        )
    body = body_path.read_text()
    if "_TODO:" in body and not force:
        return _blocked(
            state, "pr", "the PR description still has TODO sections.",
            f"Finish `{shown}` first (run `/flow pr` to have it written). "
            "A PR you cannot explain is not ready.",
        )
    if shutil.which("gh") is None:
        return FlowResult("**/flow pr:** the `gh` CLI is not installed.", state, ok=False)

    # Stage tracked changes plus new files, but never BACH's own artifacts.
    new_files = _changed_paths(root, "HEAD")[1]
    steps = [
        ["git", "add", "--update", "--", "."],
        *([["git", "add", "--", *new_files]] if new_files else []),
        ["git", "commit", "-m", title],
        ["git", "push", "-u", "origin", branch],
        ["gh", "pr", "create", "--title", title, "--body-file", str(body_path),
         "--base", state.base_branch, "--head", branch],
    ]
    out = ""
    for step in steps:
        if step[:2] == ["git", "commit"] and not _git(root, "diff", "--cached", "--name-only")[1]:
            continue  # nothing staged; the work is already committed
        code, out = _run(step, root, timeout=120.0)
        if code != 0:
            return FlowResult(
                f"**/flow pr failed** at `{' '.join(step[:3])}`:\n\n{_tail(out, 15)}",
                state, ok=False,
            )
    url = next((ln for ln in reversed(out.splitlines()) if ln.startswith("http")), out.strip())
    body_path.unlink(missing_ok=True)  # its text now lives in the pull request
    state.pr_url = url
    save_state(root, state)
    return FlowResult(f"**Pull request opened:** {url}", state)


# ---------------------------------------------------------------------------
# Reviewing someone else's pull request
# ---------------------------------------------------------------------------

_PR_REVIEWS_FILE = "bach-flow-pr-reviews.json"   # {pr number: head commit reviewed}
_PR_FIELDS = (
    "number,title,author,headRefName,baseRefName,headRefOid,isDraft,state,"
    "reviewDecision,mergeable,statusCheckRollup,files,url,body,additions,deletions"
)


def _gh_json(root: Path, *args: str) -> tuple[Optional[object], str]:
    """Run `gh ... --json`, return (parsed result, error text)."""
    if shutil.which("gh") is None:
        return None, "the GitHub CLI (`gh`) is not installed"
    code, out = _run(["gh", *args], root, timeout=60.0)
    if code != 0:
        return None, _tail(out, 6) or "gh failed"
    try:
        return json.loads(out), ""
    except ValueError:
        return None, f"could not read gh's answer: {out[:200]}"


def _reviewed_heads(root: Path) -> dict:
    git_dir = _git_dir(root)
    try:
        return json.loads((git_dir / _PR_REVIEWS_FILE).read_text()) if git_dir else {}
    except (OSError, ValueError):
        return {}


def _remember_review(root: Path, number: int, head: str) -> None:
    git_dir = _git_dir(root)
    if git_dir is None:
        return
    seen = _reviewed_heads(root)
    seen[str(number)] = head
    (git_dir / _PR_REVIEWS_FILE).write_text(json.dumps(seen, indent=2))


def _checks(pr: dict) -> tuple[str, list[str]]:
    """("passing" | "failing" | "pending" | "none", names of failing checks)."""
    runs = pr.get("statusCheckRollup") or []
    if not runs:
        return "none", []
    failing, pending = [], False
    for run in runs:
        outcome = (run.get("conclusion") or run.get("state") or "").upper()
        name = run.get("name") or run.get("context") or "check"
        if outcome in ("FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED"):
            failing.append(name)
        elif outcome in ("", "PENDING", "QUEUED", "IN_PROGRESS", "EXPECTED"):
            pending = True
    return ("failing" if failing else "pending" if pending else "passing"), failing


def _pr_line(pr: dict) -> str:
    author = (pr.get("author") or {}).get("login") or "unknown"
    decision = {
        "APPROVED": "approved", "CHANGES_REQUESTED": "changes requested",
        "REVIEW_REQUIRED": "needs review",
    }.get(pr.get("reviewDecision") or "", "not reviewed")
    draft = ", draft" if pr.get("isDraft") else ""
    return (
        f"- `/flow prs {pr['number']}` {pr.get('title', '')} "
        f"(by {author}, +{pr.get('additions', 0)} -{pr.get('deletions', 0)}, {decision}{draft})"
    )


def _pr_list(root: Path, state: FlowState) -> FlowResult:
    prs, error = _gh_json(
        root, "pr", "list", "--limit", "30", "--json",
        "number,title,author,isDraft,reviewDecision,additions,deletions",
    )
    if prs is None:
        return FlowResult(f"**/flow prs failed:** {error}", state, ok=False)
    if not prs:
        return FlowResult("There are no open pull requests.", state)
    lines = [f"**Open pull requests** ({len(prs)})", "", *(_pr_line(pr) for pr in prs)]
    lines += ["", "Reviewing one reads it; nothing is posted to GitHub until you approve or request changes."]
    return FlowResult("\n".join(lines), state)


def _pr_fetch(root: Path, number: int) -> tuple[Optional[dict], str]:
    pr, error = _gh_json(root, "pr", "view", str(number), "--json", _PR_FIELDS)
    if not isinstance(pr, dict):
        return None, error
    return pr, ""


def _pr_test_run(root: Path, state: FlowState, ref: str) -> str:
    """Run the tests on the pull request's code in a throwaway worktree, so
    the working directory and the current branch are left alone."""
    import tempfile

    scope = scope_of(root)
    holder = Path(tempfile.mkdtemp(prefix="bach-pr-"))
    tree = holder / "tree"
    code, out = _git(root, "worktree", "add", "--detach", str(tree), ref)
    if code != 0:
        shutil.rmtree(holder, ignore_errors=True)
        return f"Tests were not run: could not check the pull request out ({_tail(out, 3)})."
    try:
        where = tree / scope if scope else tree
        command = state.test_command or detect_test_command(where)
        if not command:
            return "Tests were not run: no test command is known. Run `/flow onboard` first."
        run, output = run_tests(where, command)
        line = f"Tests on the pull request's code (`{command}`): {run.summary or 'no summary'} (exit {run.returncode})."
        if run.failing:
            line += "\nFailing: " + ", ".join(run.failing[:15])
        elif run.returncode != 0:
            line += "\n\n```\n" + _tail(output, 15) + "\n```"
        return line
    finally:
        _git(root, "worktree", "remove", "--force", str(tree))
        shutil.rmtree(holder, ignore_errors=True)


def _pr_review_prompt(pr: dict, ref: str, diff: str, tests: str, checks: str) -> str:
    number = pr["number"]
    author = (pr.get("author") or {}).get("login") or "unknown"
    body = (pr.get("body") or "").strip() or "(no description)"
    if len(body) > 3000:
        body = body[:3000] + "\n[...]"
    if len(diff) > 20000:
        diff = diff[:20000] + "\n[... diff truncated; read the rest with the git commands above ...]"
    return f"""Review pull request #{number}: "{pr.get('title', '')}" by {author}
({pr.get('headRefName')} -> {pr.get('baseRefName')}).

This is someone else's work. Read only: do not check it out, do not edit,
commit or push anything, and do not post anything to GitHub.

Its code is available locally as `{ref}`:
- `git show {ref}:<path>` shows a file as the pull request has it.
- `git diff {pr.get('baseRefName')}...{ref}` is the full change.

Checks on GitHub: {checks}
{tests}

Report, in this order:

1. **What it does**: two or three sentences. The problem it addresses and
   how it fixes it, in your own words (not the author's description).
2. **Findings**, most serious first. For each: `file:line`, what is wrong,
   why it matters, and the change you would ask for. Label each Blocking,
   Should fix or Nit. Only report what you can point to in the code; if
   you found nothing in a category, say so.
3. **Tests**: is there a test that would fail without this change? What
   behaviour is changed but not tested?
4. **Verdict**: Approve or Request changes, with one sentence of reason.

Then stop. Say nothing about the workflow tool.

The author's description:

{body}

The change:

```diff
{diff}
```
"""


def _pr_review(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    """`/flow prs`: list, review, approve and merge someone else's pull request."""
    # `/flow prs`              list
    # `/flow prs 12`           review
    # `/flow prs 12 test`      review, and run the tests on its code
    # `/flow prs 12 approve [note]` | `changes <text>` | `merge [rebase|merge-commit]`
    text = args.strip()
    if not text or text == "list":
        return _pr_list(root, state)
    parsed = re.match(r"#?(\d+)(?:\s+([a-z-]+))?\s*(.*)$", text, flags=re.S)
    verbs = {"": "review", "review": "review", "test": "test", "approve": "approve",
             "changes": "changes", "merge": "merge"}
    if not parsed or (parsed.group(2) or "") not in verbs:
        return FlowResult(
            "**/flow prs:** give a pull request number, then optionally one of "
            "`test`, `approve`, `changes <text>`, `merge`. `/flow prs` alone lists them.",
            state, ok=False,
        )
    number = int(parsed.group(1))
    action = verbs[parsed.group(2) or ""]
    note = parsed.group(3).strip()
    extra = note.split()
    pr, error = _pr_fetch(root, number)
    if pr is None:
        return FlowResult(f"**/flow prs failed:** {error}", state, ok=False)
    head = pr.get("headRefOid") or ""
    status, failing = _checks(pr)
    checks = {
        "none": "none configured", "passing": "passing", "pending": "still running",
        "failing": "FAILING (" + ", ".join(failing) + ")",
    }[status]
    title = f"#{number} {pr.get('title', '')}"

    if action in ("review", "test"):
        ref = f"refs/bach/pr-{number}"
        code, out = _git(root, "fetch", "--quiet", "origin", f"+pull/{number}/head:{ref}", timeout=120.0)
        if code != 0:
            return FlowResult(
                f"**/flow prs failed:** could not fetch the pull request's code ({_tail(out, 3)}).",
                state, ok=False,
            )
        tests = _pr_test_run(root, state, ref) if action == "test" else ""
        _, diff = _git(root, "diff", f"{pr.get('baseRefName')}...{ref}")
        if not diff:
            _, diff = _git(root, "diff", f"origin/{pr.get('baseRefName')}...{ref}")
        files = sorted(
            (f.get("path", ""), int(f.get("additions") or 0), int(f.get("deletions") or 0), False)
            for f in pr.get("files") or []
        )
        _remember_review(root, number, head)
        author = (pr.get("author") or {}).get("login") or "unknown"
        lines = [
            f"**Reviewing pull request {title}**", "",
            f"- By: {author}",
            f"- Branch: `{pr.get('headRefName')}` -> `{pr.get('baseRefName')}`",
            f"- Checks on GitHub: {checks}",
            f"- Link: {pr.get('url', '')}",
            "", _files_text(files),
        ]
        if tests:
            lines += ["", tests]
        elif status == "none":
            lines += ["", f"No checks ran on GitHub. To run the tests on this code locally: `/flow prs {number} test`"]
        lines += [
            "", "Nothing has been posted to GitHub. After reading the review:", "",
            f"- `/flow prs {number} approve`",
            f"- `/flow prs {number} changes <what to change>`",
        ]
        return FlowResult(
            "\n".join(lines), state,
            agent_prompt=_pr_review_prompt(pr, ref, diff, tests, checks),
        )

    if pr.get("state") and pr.get("state") != "OPEN":
        return FlowResult(f"Pull request {title} is {str(pr.get('state')).lower()}, not open.", state, ok=False)
    reviewed = _reviewed_heads(root).get(str(number))

    if action in ("approve", "changes"):
        if not force and reviewed != head:
            why = (
                "it has new commits since you reviewed it" if reviewed
                else "it has not been reviewed here"
            )
            return FlowResult(
                f"**Blocked:** pull request {title} cannot be answered yet: {why}.\n\n"
                f"**Do next:** `/flow prs {number}`", state, ok=False,
            )
        if action == "changes" and not note:
            return FlowResult(
                "**/flow prs:** asking for changes needs the text to send to the "
                f"author: `/flow prs {number} changes <what to change>`",
                state, ok=False,
            )
        if action == "approve" and status == "failing" and not force:
            return FlowResult(
                f"**Blocked:** checks are failing on {title} ({', '.join(failing)}). "
                "Approving a red pull request needs `--force`.", state, ok=False,
            )
        command = ["gh", "pr", "review", str(number), "--approve" if action == "approve" else "--request-changes"]
        if note:
            command += ["--body", note]
        code, out = _run(command, root, timeout=60.0)
        if code != 0:
            return FlowResult(f"**GitHub refused:** {_tail(out, 5)}", state, ok=False)
        if action == "approve":
            return FlowResult(
                f"**Approved** pull request {title} on GitHub.\n\n"
                f"**Do next:** `/flow prs {number} merge` when it should go in.",
                state,
            )
        return FlowResult(f"**Changes requested** on pull request {title}. The author has your notes.", state)

    # --merge
    problems = []
    if pr.get("isDraft"):
        problems.append("it is still a draft")
    if (pr.get("reviewDecision") or "") != "APPROVED":
        problems.append("it has not been approved")
    if status == "failing":
        problems.append("checks are failing (" + ", ".join(failing) + ")")
    if status == "pending":
        problems.append("checks are still running")
    if pr.get("mergeable") == "CONFLICTING":
        problems.append("it conflicts with the base branch")
    if problems and not force:
        return FlowResult(
            f"**Blocked:** pull request {title} is not ready to merge: " + "; ".join(problems) + ".\n\n"
            f"`/flow prs {number} merge --force` overrides this.", state, ok=False,
        )
    method = "--rebase" if "rebase" in extra else "--merge" if "merge-commit" in extra else "--squash"
    code, out = _run(["gh", "pr", "merge", str(number), method], root, timeout=120.0)
    if code != 0:
        return FlowResult(f"**GitHub refused the merge:** {_tail(out, 5)}", state, ok=False)
    return FlowResult(
        f"**Merged** pull request {title} into `{pr.get('baseRefName')}` ({method.lstrip('-')}).",
        state,
    )


def _status(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    command, where = next_step(state, root)
    current = state.current()
    number = PHASE_NAMES.index(current) + 1
    lines = [
        f"**You are in: {current}** (step {number} of {len(PHASE_NAMES)})",
        "",
        where,
        "",
        f"**Do next:** `{command}`",
        "",
        "```", checklist(state), "```",
    ]
    if scope_of(root):
        lines.append(f"- Scope: `{scope_of(root)}` (change with `/flow scope`)")
    if state.issue_title:
        lines.append(f"- Issue: {state.issue_title}")
    if state.branch:
        lines.append(f"- Branch: `{state.branch}`")
    if state.baseline:
        lines.append(f"- Baseline: {state.baseline.get('summary')}")
    if state.repro_target:
        lines.append(f"- Regression test: `{state.repro_target}`")
    if state.verified and _stale(root, state):
        lines.append("- **The code changed since `/flow verify`.**")
    if state.reviewed_by:
        lines.append(f"- Reviewed by: {state.reviewed_by}")
    if state.pr_url:
        lines.append(f"- PR: {state.pr_url}")
    return FlowResult("\n".join(lines), state)


def _reset(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    path = _state_path(root)
    if path.exists():
        path.unlink()
    return FlowResult("Workflow state cleared.", FlowState())


def _scope_summary(top: Path, scope: str) -> str:
    state = load_state(top / scope if scope else top)
    if state.baseline is None and not state.issue:
        return "not started"
    what = f"{state.current()}"
    return f"{what}: {state.issue_title}" if state.issue_title else what


def _scope_command(path: Path, args: str) -> FlowResult:
    """`/flow scope [dir | --clear]`: choose the directory to work in.

    A repository can hold many projects. Each directory has its own saved
    workflow state, and its tests and diff are limited to that directory,
    so work in one does not disturb another and nothing needs copying out.
    """
    top = _toplevel(path)
    if top is None:
        return FlowResult("**/flow scope:** not inside a git repository.", FlowState(), ok=False)
    git_dir = _git_dir(top)
    if path != top:
        here = scope_of(path)
        return FlowResult(
            f"This session was started inside `{here}`, so that directory is "
            "the scope. To choose a different one, start the session at the "
            f"top of the repository (`{top}`) and run `/flow scope <dir>`.",
            load_state(path), ok=not args.strip(),
        )
    arg = args.strip()
    if arg in ("--clear", "."):
        (git_dir / _SCOPE_FILE).unlink(missing_ok=True)
        return FlowResult(
            "Scope cleared: the workflow now covers the whole repository.\n\n"
            + _status(top, load_state(top), "", False).message,
            load_state(top),
        )
    if arg:
        target = (top / arg).resolve()
        if not target.is_dir() or top not in (target, *target.parents):
            return FlowResult(
                f"**/flow scope:** `{arg}` is not a directory in this repository. "
                "Give a path relative to the top of the repository.",
                load_state(project_root(top)), ok=False,
            )
        if target == top:
            (git_dir / _SCOPE_FILE).unlink(missing_ok=True)
        else:
            (git_dir / _SCOPE_FILE).write_text(target.relative_to(top).as_posix() + "\n")
        root = project_root(top)
        return FlowResult(
            f"**Scope set to `{scope_of(root) or '.'}`.** Every flow command now "
            "works in that directory: its tests, its diff, its own saved progress.\n\n"
            + _status(root, load_state(root), "", False).message,
            load_state(root),
        )

    current = _chosen_scope(top)
    known = {""}
    scope_dir = git_dir / _SCOPE_DIR
    if scope_dir.is_dir():
        suffix = f".{_STATE_FILE}"
        known |= {
            f.name[: -len(suffix)].replace("__", "/")
            for f in scope_dir.iterdir() if f.name.endswith(suffix)
        }
    known = {k for k in known if (top / k).is_dir()} | {current}
    lines = [f"**Scope: `{current or '.'}`**" + ("" if current else " (the whole repository)"), ""]
    for scope in sorted(known):
        mark = "→" if scope == current else " "
        command = f"/flow scope {scope}" if scope else "/flow scope --clear"
        lines.append(f"- {mark} `{command}` {_scope_summary(top, scope)}")
    lines += ["", "Work in another directory: `/flow scope <dir>` (path from the top of the repository)."]
    return FlowResult("\n".join(lines), load_state(project_root(top)))


def _scope_note(root: Path) -> str:
    """Tells the model where it is working. Without this an agent started
    at the top of a large repository reads, runs and edits anywhere."""
    scope = scope_of(root)
    if not scope:
        return ""
    return (
        f"Scope: this work is limited to the directory `{scope}` inside a larger "
        f"repository (absolute path `{root}`). Run every command from that "
        "directory; paths in this task are relative to it. Do not read, run "
        "or change anything outside it.\n\n"
    )


_HELP = [
    ("The seven steps, in order", [
        ("/flow onboard [test command]", "Map the repo and record the test baseline"),
        ("/flow issue [file | #12 | ENG-123 | text]", "Pick an issue (no argument lists them) and trace its cause"),
        ("/flow feature [ENG-123 | file | text]", "Step 2 for a feature: plan against its spec, post the plan to the issue"),
        ("/flow repro", "Ask for one failing test (a feature: tests from its acceptance criteria)"),
        ("/flow repro <test id>", "Run that test: it has to fail"),
        ("/flow fix [which findings]", "Smallest correct fix; after a failed verify or a review, addresses that"),
        ("/flow verify", "Regression test passes and nothing else broke"),
        ("/flow review", "Independent review of your own change"),
        ("/flow pr", "Write the pull request description (PR-DRAFT.md)"),
        ("/flow pr --diff", "Changed files with line counts, then each diff"),
        ("/flow pr --create", "Commit, push and open the pull request"),
    ]),
    ("Any time", [
        ("/flow status", "Where am I, and what do I do next"),
        ("/flow scope [dir | --clear]", "Work in one directory of a large repository (no argument lists them)"),
        ("/flow linear [project]", "List Linear issues to pick from"),
        ("/flow reset", "Forget the saved progress for this directory"),
        ("/flow help", "This list"),
    ]),
    ("Someone else's pull request", [
        ("/flow prs", "List open pull requests"),
        ("/flow prs 12", "Review number 12 (nothing is posted)"),
        ("/flow prs 12 test", "Review it and run the tests on its code"),
        ("/flow prs 12 approve", "Approve it on GitHub"),
        ("/flow prs 12 changes <text>", "Ask the author for changes"),
        ("/flow prs 12 merge", "Merge it (needs approval and passing checks)"),
    ]),
]


def help_text(extra: Optional[list[tuple[str, list[tuple[str, str]]]]] = None) -> str:
    """Every command, grouped, as aligned plain text. EXTRA adds groups a
    surface knows about (commands installed alongside these)."""
    groups = _HELP + list(extra or [])
    width = max(len(command) for _, rows in groups for command, _ in rows)
    lines = []
    for title, rows in groups:
        lines += [title, *(f"  {command:<{width}}  {what}" for command, what in rows), ""]
    lines.append("--force overrides a blocked step. Each step refuses to run until the one before it is done.")
    return "```\n" + "\n".join(lines) + "\n```"


def _help(root: Path, state: FlowState, args: str, force: bool, provider: str = "") -> FlowResult:
    command, _ = next_step(state, root)
    where = f"You are in: **{state.current()}**. Do next: `{command}`"
    if scope_of(root):
        where += f" (scope: `{scope_of(root)}`)"
    return FlowResult(help_text() + "\n\n" + where, state)


_PHASE_FUNCS = {
    "status": _status,
    "help": _help,
    "--help": _help,
    "-h": _help,
    "onboard": _onboard,
    "issue": _issue,
    "feature": _feature,
    "linear": _linear_phase,
    "repro": _repro,
    "fix": _fix,
    "verify": _verify,
    "review": _review,
    "pr": _pr,
    "prs": _pr_review,
    "reset": _reset,
}

USAGE = (
    "`/flow <status | scope [dir] | onboard [test cmd] | issue <text|file|#n|ENG-123> | feature <text|file|ENG-123> | "
    "linear [project] | repro [test id] "
    "| fix | verify | review [--with <provider>] | pr [--diff | --create] | "
    "prs [N [test | approve | changes <text> | merge]] | reset>`"
)


def run_flow(
    project_path: str | Path, argline: str = "", *, provider: str = ""
) -> FlowResult:
    """Entry point for every surface. ARGLINE is whatever followed `/flow`.

    PROVIDER is the name of the surface's active reasoning model; it is
    recorded as the author when a fix is requested.
    """
    head, _, rest = argline.strip().partition(" ")
    sub = head or "status"
    if sub == "scope":
        return _scope_command(Path(project_path).expanduser().resolve(), rest)
    root = project_root(project_path)
    state = load_state(root)
    func = _PHASE_FUNCS.get(sub)
    if func is None:
        return FlowResult(f"Unknown `/flow {sub}`. Usage: {USAGE}", state, ok=False)
    tokens = rest.split()
    force = "--force" in tokens
    if force:
        rest = " ".join(t for t in tokens if t != "--force")
    result = func(root, state, rest, force, provider)
    # A pull request can touch any part of the repository, so reviewing one
    # is not confined to the scoped directory.
    note = "" if sub == "prs" else _scope_note(root)
    if note:
        if result.agent_prompt:
            result.agent_prompt = note + result.agent_prompt
        if result.review_prompt:
            result.review_prompt = note + result.review_prompt
    return result


__all__ = [
    "PHASES",
    "FlowResult",
    "FlowState",
    "acceptance_criteria",
    "changed_files",
    "checklist",
    "detect_test_command",
    "find_issues",
    "help_text",
    "diff_fingerprint",
    "load_state",
    "next_step",
    "parse_test_output",
    "plan_entries",
    "pr_body",
    "project_root",
    "record_plan",
    "record_review",
    "review_base",
    "scope_of",
    "working_diff",
    "run_flow",
]
