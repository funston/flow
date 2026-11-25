"""Unit tests for `flow.workflow` — the gated `/flow` workflow shared by
the CLI and ACP surfaces.

No mocks: every test builds a real git repository with a real pytest suite in
`tmp_path` and drives `run_flow` against it. The reasoning model is not
involved; "the agent wrote a test / a fix" is simulated by writing the file.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from flow import workflow as wf  # noqa: E402

BUGGY = "def clamp(x, lo, hi):\n    return max(lo, min(x, lo))\n"
# Different length from BUGGY on purpose: CPython keys cached bytecode on
# (mtime, size), and these tests rewrite the file within the same second.
FIXED = "def clamp(x, lo, hi):\n    upper = min(x, hi)\n    return max(lo, upper)\n"
EXISTING_TEST = "from calc import clamp\n\ndef test_low():\n    assert clamp(-5, 0, 10) == 0\n"
REPRO_TEST = "from calc import clamp\n\ndef test_inside_range():\n    assert clamp(5, 0, 10) == 5\n"
REPRO_ID = "tests/test_repro.py::test_inside_range"


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "calc.py").write_text(BUGGY)
    (tmp_path / "tests" / "test_calc.py").write_text(EXISTING_TEST)
    (tmp_path / "README.md").write_text("# calc\n")
    # Isolate the inner pytest from plugins installed around the outer one.
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\npythonpath = .\naddopts = -p no:cacheprovider\n"
    )
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "init")
    return tmp_path


@pytest.fixture(autouse=True)
def _no_outer_plugins(monkeypatch):
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def to_red(repo: Path) -> None:
    assert wf.run_flow(repo, "onboard").ok
    assert wf.run_flow(repo, "issue clamp returns lo for in-range values").ok
    (repo / "tests" / "test_repro.py").write_text(REPRO_TEST)
    assert wf.run_flow(repo, f"repro {REPRO_ID}").ok


def do_review(repo: Path, args: str = "", by: str = "claude (test)") -> wf.FlowResult:
    """`/flow review`, with the surface's half (the reviewer agent finishing)
    simulated by `record_review`."""
    result = wf.run_flow(repo, f"review {args}".strip())
    if result.review_prompt:
        wf.record_review(repo, result.review_fingerprint, by)
    return result


# --- pure helpers ---------------------------------------------------------


def test_parse_pytest_summary():
    parsed = wf.parse_test_output("....\n=== 3 failed, 12 passed, 1 error in 0.4s ===\n")
    assert (parsed["passed"], parsed["failed"], parsed["errors"]) == (12, 3, 1)


def test_parse_all_green_fills_zeroes():
    parsed = wf.parse_test_output("13 passed in 0.06s")
    assert (parsed["passed"], parsed["failed"], parsed["errors"]) == (13, 0, 0)


def test_parse_unknown_runner_leaves_counts_unset():
    parsed = wf.parse_test_output("ok  \tpkg/foo\t0.01s")
    assert parsed["passed"] is None and parsed["summary"].startswith("ok")


def test_detects_pytest(repo):
    assert wf.detect_test_command(repo).endswith("-m pytest -q")


def test_detects_nothing_in_empty_dir(tmp_path):
    assert wf.detect_test_command(tmp_path) == ""


# --- state ----------------------------------------------------------------


def test_state_lives_in_git_dir_and_is_never_tracked(repo):
    wf.run_flow(repo, "onboard")
    assert (repo / ".git" / "flow.json").exists()
    assert git(repo, "status", "--porcelain") == ""


def test_status_on_fresh_repo_points_at_onboard(repo):
    result = wf.run_flow(repo, "")
    assert "[>] onboard" in result.message
    assert result.agent_prompt is None


# --- phases and gates -----------------------------------------------------


def test_onboard_records_baseline_and_asks_for_recon(repo):
    result = wf.run_flow(repo, "onboard")
    assert result.ok
    assert result.state.baseline["passed"] == 1
    assert result.state.base_branch == "main"
    assert "Inspect it without modifying" in result.agent_prompt


def test_issue_requires_baseline(repo):
    result = wf.run_flow(repo, "issue something is wrong")
    assert not result.ok and "blocked" in result.message


def test_issue_creates_branch_and_investigates_read_only(repo):
    wf.run_flow(repo, "onboard")
    result = wf.run_flow(repo, "issue clamp returns lo for in-range values")
    assert result.ok
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "fix/clamp-returns-lo-for-in-range-values"
    assert "Do not modify any code" in result.agent_prompt


def test_issue_reads_text_from_a_file(repo):
    (repo / "ISSUE.md").write_text("# Clamp is broken\n\nDetails here.\n")
    wf.run_flow(repo, "onboard")
    result = wf.run_flow(repo, "issue ISSUE.md")
    assert result.state.issue_title == "Clamp is broken"
    assert "Details here." in result.agent_prompt


def test_fix_is_blocked_without_a_red_test(repo):
    wf.run_flow(repo, "onboard")
    wf.run_flow(repo, "issue clamp bug")
    result = wf.run_flow(repo, "fix")
    assert not result.ok and result.agent_prompt is None


def test_fix_force_overrides_the_gate(repo):
    wf.run_flow(repo, "onboard")
    wf.run_flow(repo, "issue clamp bug")
    assert wf.run_flow(repo, "fix --force").agent_prompt


def test_repro_without_target_asks_agent_for_a_test(repo):
    wf.run_flow(repo, "onboard")
    wf.run_flow(repo, "issue clamp bug")
    result = wf.run_flow(repo, "repro")
    assert "regression test" in result.agent_prompt
    assert result.state.repro_red is None


def test_repro_rejects_a_test_that_passes(repo):
    wf.run_flow(repo, "onboard")
    wf.run_flow(repo, "issue clamp bug")
    result = wf.run_flow(repo, "repro tests/test_calc.py::test_low")
    assert not result.ok and "passed" in result.message
    assert result.state.repro_red is None


def test_repro_rejects_a_test_id_that_does_not_exist(repo):
    wf.run_flow(repo, "onboard")
    wf.run_flow(repo, "issue clamp bug")
    result = wf.run_flow(repo, "repro tests/test_nope.py")
    assert not result.ok and result.state.repro_red is None


def test_repro_accepts_a_red_test_and_unlocks_fix(repo):
    to_red(repo)
    state = wf.load_state(repo)
    assert state.repro_target == REPRO_ID and state.repro_red["failed"] == 1
    result = wf.run_flow(repo, "fix")
    assert result.ok and REPRO_ID in result.agent_prompt


def test_verify_fails_while_the_bug_is_still_there(repo):
    to_red(repo)
    wf.run_flow(repo, "fix")
    result = wf.run_flow(repo, "verify")
    assert not result.ok and "STILL RED" in result.message
    assert result.state.verified is None


def test_verify_catches_a_regression_against_the_baseline(repo):
    to_red(repo)
    wf.run_flow(repo, "fix")
    # A "fix" that makes the regression test pass but breaks existing behaviour.
    (repo / "calc.py").write_text("def clamp(x, lo, hi):\n    return x\n")
    result = wf.run_flow(repo, "verify")
    assert not result.ok and "fail that did not fail before" in result.message
    assert "tests/test_calc.py::" in result.message  # named, not counted


def test_verify_catches_a_deleted_test(repo):
    to_red(repo)
    wf.run_flow(repo, "fix")
    (repo / "calc.py").write_text(FIXED)
    (repo / "tests" / "test_calc.py").unlink()
    result = wf.run_flow(repo, "verify")
    # The suite is green, but an existing test file is gone: checked directly.
    assert not result.ok and "tests/test_calc.py" in result.message
    assert "deleted" in result.message


def test_review_is_blocked_until_verified(repo):
    to_red(repo)
    result = wf.run_flow(repo, "review")
    assert not result.ok


def test_full_happy_path_to_pr_draft(repo):
    to_red(repo)
    wf.run_flow(repo, "fix")
    (repo / "calc.py").write_text(FIXED)

    verified = wf.run_flow(repo, "verify")
    assert verified.ok and verified.state.verified["passed"] == 2

    review = do_review(repo)
    assert review.ok
    assert review.agent_prompt is None                   # never the author agent
    assert "upper = min(x, hi)" in review.review_prompt  # tracked change
    assert "test_inside_range" in review.review_prompt   # untracked new file
    assert "clamp returns lo" in review.review_prompt    # original issue
    assert git(repo, "diff", "--cached", "--name-only") == ""  # index untouched

    draft = wf.run_flow(repo, "pr")
    assert draft.ok
    body = (repo / "PR-DRAFT.md").read_text()
    assert "1 failed" in body and "2 passed" in body and "_TODO:" in body
    assert "| Reviewed by | claude (test) |" in body
    assert git(repo, "log", "--oneline").count("\n") == 0  # nothing committed

    blocked = wf.run_flow(repo, "pr --create")
    assert not blocked.ok and "TODO" in blocked.message


def test_editing_after_verify_blocks_review_and_pr(repo):
    to_red(repo)
    wf.run_flow(repo, "fix")
    (repo / "calc.py").write_text(FIXED)
    assert wf.run_flow(repo, "verify").ok
    (repo / "calc.py").write_text(FIXED + "\n# one more tweak\n")
    result = wf.run_flow(repo, "review")
    assert not result.ok and "changed after" in result.message


def test_committing_verified_work_does_not_invalidate_it(repo):
    to_red(repo)
    wf.run_flow(repo, "fix")
    (repo / "calc.py").write_text(FIXED)
    assert wf.run_flow(repo, "verify").ok
    assert do_review(repo).ok
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "fix clamp")
    assert wf.run_flow(repo, "pr").ok


def test_review_is_not_done_until_a_reviewer_has_finished(repo):
    to_red(repo)
    wf.run_flow(repo, "fix")
    (repo / "calc.py").write_text(FIXED)
    wf.run_flow(repo, "verify")
    requested = wf.run_flow(repo, "review")
    assert requested.review_prompt and not requested.state.done("review")
    blocked = wf.run_flow(repo, "pr")
    assert not blocked.ok and "not been reviewed" in blocked.message
    wf.record_review(repo, requested.review_fingerprint, "claude (test)")
    assert wf.run_flow(repo, "pr").ok


def test_review_with_selects_and_remembers_the_reviewer(repo):
    to_red(repo)
    wf.run_flow(repo, "fix", provider="local")
    (repo / "calc.py").write_text(FIXED)
    wf.run_flow(repo, "verify")
    first = wf.run_flow(repo, "review --with claude")
    assert first.reviewer == "claude" and first.state.author == "local"
    assert wf.run_flow(repo, "review").reviewer == "claude"


def test_review_with_needs_a_provider_name(repo):
    to_red(repo)
    wf.run_flow(repo, "fix")
    (repo / "calc.py").write_text(FIXED)
    wf.run_flow(repo, "verify")
    assert not wf.run_flow(repo, "review --with").ok


def test_the_tools_own_files_do_not_invalidate_or_enter_the_review(repo):
    to_red(repo)
    wf.run_flow(repo, "fix")
    (repo / "calc.py").write_text(FIXED)
    assert wf.run_flow(repo, "verify").ok
    # What the shell's Ranger and transcript capture leave in a project.
    (repo / ".ranger.json").write_text("{}")
    (repo / ".ranger_stats").write_text("{}")
    (repo / "session").mkdir()
    (repo / "session" / "_history.jsonl").write_text("{}\n")
    review = do_review(repo)
    assert review.ok and ".ranger" not in review.review_prompt
    assert wf.run_flow(repo, "pr").ok
    assert ".ranger" not in (repo / "PR-DRAFT.md").read_text()


def test_pr_create_commits_the_change_without_tool_artifacts(repo, tmp_path_factory):
    remote = tmp_path_factory.mktemp("remote")
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    git(repo, "remote", "add", "origin", str(remote))
    to_red(repo)
    wf.run_flow(repo, "fix")
    (repo / "calc.py").write_text(FIXED)
    wf.run_flow(repo, "verify")
    do_review(repo)
    (repo / ".ranger.json").write_text("{}")
    wf.run_flow(repo, "pr")
    body = repo / "PR-DRAFT.md"
    body.write_text(body.read_text().replace("_TODO:", "done:"))

    result = wf.run_flow(repo, "pr --create")

    # Commit and push are real. `gh pr create` cannot succeed against a
    # local bare remote, so the run stops there and says so.
    assert not result.ok and "gh pr create" in result.message
    committed = git(repo, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert sorted(committed) == ["calc.py", "tests/test_repro.py"]
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    assert git(repo, "ls-remote", "--heads", "origin", branch)
    assert wf.load_state(repo).pr_url == ""


def test_same_size_edit_right_after_a_test_run_is_not_masked_by_stale_bytecode(repo):
    wf.run_flow(repo, "onboard")                    # runs the suite, imports calc
    wf.run_flow(repo, "issue clamp bug")
    (repo / "tests" / "test_repro.py").write_text(REPRO_TEST)
    wf.run_flow(repo, f"repro {REPRO_ID}")
    wf.run_flow(repo, "fix")
    same_size = BUGGY.replace("min(x, lo)", "min(x, hi)")
    assert len(same_size) == len(BUGGY)
    (repo / "calc.py").write_text(same_size)
    assert wf.run_flow(repo, "verify").ok


def test_new_issue_resets_downstream_phases_but_keeps_baseline(repo):
    to_red(repo)
    # Switching issues part-way through is deliberate, so it needs --force.
    result = wf.run_flow(repo, "issue a different bug --force")
    assert result.state.repro_red is None
    assert result.state.baseline["passed"] == 1


def test_checklist_and_plan_track_progress(repo):
    to_red(repo)
    state = wf.load_state(repo)
    assert "[x] repro" in wf.checklist(state) and "[>] fix" in wf.checklist(state)
    statuses = dict(
        (content.split(":")[0], status) for content, status in wf.plan_entries(state)
    )
    assert statuses["repro"] == "completed"
    assert statuses["fix"] == "in_progress"
    assert statuses["pr"] == "pending"


def test_unknown_subcommand(repo):
    result = wf.run_flow(repo, "frobnicate")
    assert not result.ok and "Usage" in result.message


def test_reset_clears_state(repo):
    wf.run_flow(repo, "onboard")
    wf.run_flow(repo, "reset")
    assert wf.load_state(repo).baseline is None


def test_tests_run_on_the_users_python_not_the_tools(tmp_path, monkeypatch):
    """A project with no virtualenv must be tested with the Python on PATH
    (where its dependencies are installed), not the interpreter flow runs in.
    Seen live: `/flow onboard` failed with "No module named 'torch'"."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    shell_python = fake_bin / "python3"
    shell_python.write_text("#!/bin/sh\n")
    shell_python.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    project = tmp_path / "proj"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")

    assert wf.detect_test_command(project).startswith(str(shell_python))

    # A project virtualenv still wins.
    venv_python = project / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    assert wf.detect_test_command(project).startswith(str(venv_python))


def test_onboard_refuses_a_baseline_when_the_tests_cannot_start(tmp_path):
    """A missing dependency makes pytest exit before running anything. That
    must not be recorded as the baseline, and the guessed command must not
    stick, or every later phase compares against a suite that never ran."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text("import module_that_is_not_installed\n")
    (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")

    result = wf.run_flow(tmp_path, "onboard")

    assert not result.ok and result.agent_prompt is None
    assert "could not start" in result.message
    assert "module_that_is_not_installed" in result.message
    state = wf.load_state(tmp_path)
    assert not state.done("onboard")
    assert state.test_command == ""


def test_onboard_recovers_from_a_saved_baseline_that_never_started(tmp_path):
    """State written before the could-not-start check must not pin the bad
    command forever: the next onboard re-detects and records a real baseline."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    state = wf.FlowState(
        test_command="/nonexistent/python -m pytest -q",
        baseline={"command": "/nonexistent/python -m pytest -q", "returncode": 4,
                  "passed": None, "failed": None, "errors": None,
                  "summary": "ModuleNotFoundError", "seconds": 0.1},
    )
    wf.save_state(tmp_path, state)

    result = wf.run_flow(tmp_path, "onboard")

    assert result.ok
    fresh = wf.load_state(tmp_path)
    assert "/nonexistent/python" not in fresh.test_command
    assert fresh.baseline["passed"] == 1


def _repo_with_issue_files(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    issues = tmp_path / "issues"
    issues.mkdir()
    (issues / "001-wrong-total.md").write_text("# 001: Totals are off by one\n\nDetails.\n")
    (issues / "002-slow-export.md").write_text("# 002: Export is slow\n\nDetails.\n")
    (issues / "README.md").write_text("# About these issues\n")
    return tmp_path


def test_issue_with_no_argument_lists_the_issues_to_pick_from(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # no gh: local files only
    repo = _repo_with_issue_files(tmp_path)

    result = wf.run_flow(repo, "issue")

    assert not result.ok and result.agent_prompt is None
    assert "`/flow issue issues/001-wrong-total.md` 001: Totals are off by one" in result.message
    assert "`/flow issue issues/002-slow-export.md` 002: Export is slow" in result.message
    assert "README" not in result.message
    assert not wf.load_state(repo).issue  # listing selects nothing


def test_issue_accepts_a_number_or_a_word_from_the_file_name(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    repo = _repo_with_issue_files(tmp_path)
    wf.save_state(repo, wf.FlowState(baseline={"returncode": 0, "passed": 1}))

    by_number = wf.run_flow(repo, "issue 2")
    assert by_number.ok and by_number.state.issue_title == "002: Export is slow"

    by_word = wf.run_flow(repo, "issue total")
    assert by_word.ok and by_word.state.issue_title == "001: Totals are off by one"


def test_status_says_where_we_are_and_the_one_thing_to_do_next(tmp_path):
    """Phases have two halves (ask the model, then check it ourselves), so
    status must name the exact next command at every point, and the pinned
    plan must carry it too."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)

    def status(state):
        wf.save_state(tmp_path, state)
        result = wf.run_flow(tmp_path, "status")
        current = [c for c, s in wf.plan_entries(state, tmp_path) if s == "in_progress"]
        return result.message, current[0]

    state = wf.FlowState()
    message, plan = status(state)
    assert "**You are in: onboard** (step 1 of 7)" in message
    assert "**Do next:** `/flow onboard`" in message
    assert plan.endswith("→ next: /flow onboard")

    state.baseline = {"returncode": 0, "passed": 13, "summary": "13 passed"}
    assert "**Do next:** `/flow issue`" in status(state)[0]

    state.issue, state.issue_title = "Totals are wrong", "Totals are wrong"
    message, _ = status(state)
    assert "**You are in: repro** (step 3 of 7)" in message
    assert "**Do next:** `/flow repro`" in message

    # Asked for a test, but it has not been run: still repro, different action.
    state.repro_requested = True
    message, plan = status(state)
    assert "**You are in: repro**" in message
    assert "**Do next:** `/flow repro <test id>`" in message
    assert "nobody has run it yet" in message
    assert plan.endswith("→ next: /flow repro <test id>")

    state.repro_target = "tests/test_x.py::test_total"
    state.repro_red = {"returncode": 1, "failed": 1, "summary": "1 failed"}
    message, _ = status(state)
    assert "**You are in: fix**" in message and "**Do next:** `/flow fix`" in message
    assert "tests/test_x.py::test_total" in message

    state.fix_requested = True
    message, _ = status(state)
    assert "**You are in: verify**" in message and "**Do next:** `/flow verify`" in message
    assert "unproven" in message


def test_asking_for_a_repro_is_remembered(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    wf.save_state(tmp_path, wf.FlowState(
        baseline={"returncode": 0, "passed": 1, "summary": "1 passed"},
        issue="Something is wrong", issue_title="Something is wrong",
    ))
    assert wf.run_flow(tmp_path, "repro").agent_prompt
    assert wf.next_step(wf.load_state(tmp_path), tmp_path)[0] == "/flow repro <test id>"


def test_failed_verify_names_the_tests_and_points_back_to_fix(tmp_path):
    """Seen live: a fix that addressed one of two causes. Verify must say
    which tests fail, must not suggest a test was deleted when failures
    explain the count, and status must send the user back to fix."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    assert wf.run_flow(tmp_path, "onboard").ok
    assert wf.run_flow(tmp_path, "issue the total is wrong").ok
    # Two new tests, both still failing after the "fix".
    (tests / "test_bug.py").write_text(
        "def test_bug():\n    assert 1 == 2\n\ndef test_bug_again():\n    assert 1 == 2\n"
    )
    assert wf.run_flow(tmp_path, "repro tests/test_bug.py::test_bug").ok
    first = wf.run_flow(tmp_path, "fix")
    assert "PREVIOUS FIX ATTEMPT" not in first.agent_prompt

    result = wf.run_flow(tmp_path, "verify")

    assert not result.ok
    assert "the regression test still fails" in result.message
    assert "tests/test_bug.py::test_bug" in result.message
    assert "tests/test_bug.py::test_bug_again" in result.message
    assert "deleted or skipped" not in result.message
    state = wf.load_state(tmp_path)
    command, where = wf.next_step(state, tmp_path)
    assert command == "/flow fix" and "last verify failed" in where

    second = wf.run_flow(tmp_path, "fix")
    assert "PREVIOUS FIX ATTEMPT" in second.agent_prompt
    assert "the regression test still fails" in second.agent_prompt
    assert wf.next_step(wf.load_state(tmp_path), tmp_path)[0] == "/flow verify"


def _mid_issue(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "issues").mkdir()
    (tmp_path / "issues" / "001-total.md").write_text("# 001: Total is wrong\n")
    (tmp_path / "issues" / "002-slow.md").write_text("# 002: Export is slow\n")
    assert wf.run_flow(tmp_path, "onboard").ok
    assert wf.run_flow(tmp_path, "issue issues/001-total.md").ok
    return tmp_path


def test_rerunning_the_same_issue_changes_nothing(tmp_path):
    """Seen live: re-running `/flow issue` for the issue in progress reset the
    workflow and led to a second, duplicate regression test."""
    repo = _mid_issue(tmp_path)
    assert wf.run_flow(repo, "repro").agent_prompt  # test requested
    before = wf.load_state(repo)

    again = wf.run_flow(repo, "issue issues/001-total.md")

    assert not again.ok and again.agent_prompt is None
    assert "Already working on this issue" in again.message
    assert "`/flow repro <test id>`" in again.message
    assert wf.load_state(repo) == before


def test_switching_issue_midway_needs_force(tmp_path):
    repo = _mid_issue(tmp_path)
    assert wf.run_flow(repo, "repro").agent_prompt

    other = wf.run_flow(repo, "issue issues/002-slow.md")
    assert not other.ok and "part-way through" in other.message
    assert wf.load_state(repo).issue_title == "001: Total is wrong"

    forced = wf.run_flow(repo, "issue issues/002-slow.md --force")
    assert forced.ok and forced.state.issue_title == "002: Export is slow"
    assert not forced.state.repro_requested


def test_repro_is_not_requested_twice(tmp_path):
    repo = _mid_issue(tmp_path)
    assert wf.run_flow(repo, "repro").agent_prompt

    second = wf.run_flow(repo, "repro")
    assert not second.ok and second.agent_prompt is None
    assert "already requested" in second.message

    (repo / "tests" / "test_bug.py").write_text("def test_bug():\n    assert 1 == 2\n")
    assert wf.run_flow(repo, "repro tests/test_bug.py::test_bug").ok
    third = wf.run_flow(repo, "repro")
    assert not third.ok and third.agent_prompt is None
    assert "already recorded" in third.message and "tests/test_bug.py::test_bug" in third.message

    forced = wf.run_flow(repo, "repro --force")
    assert forced.agent_prompt and "Do not leave two tests" in forced.agent_prompt


def test_onboard_midway_does_not_move_the_baseline(tmp_path):
    """Seen live: onboard was run again after the regression test and fix
    existed. The baseline went from 13 to 14 passed, so a correct fix failed
    verify with "1 fewer passing test than expected"."""
    repo = _mid_issue(tmp_path)
    (repo / "tests" / "test_bug.py").write_text("import mod\n\ndef test_bug():\n    assert mod.f() == 2\n")
    (repo / "mod.py").write_text("def f():\n    return 1\n")
    assert wf.run_flow(repo, "repro").agent_prompt
    assert wf.run_flow(repo, "repro tests/test_bug.py::test_bug").ok
    assert wf.run_flow(repo, "fix").ok
    (repo / "mod.py").write_text("def f():\n    return 2\n")

    again = wf.run_flow(repo, "onboard")

    assert not again.ok and again.agent_prompt is None
    assert "Onboard skipped" in again.message and "1 passed" in again.message
    assert wf.load_state(repo).baseline["passed"] == 1
    assert wf.run_flow(repo, "verify").ok


def test_verify_compares_tests_by_name_not_by_count(tmp_path):
    """Seen live: the baseline was re-recorded with the regression test
    already present, so "baseline + 1" could never be reached and a correct
    fix was refused. Verify now asks only: is the regression test green,
    and does anything fail that did not fail before?"""
    repo = _mid_issue(tmp_path)
    (repo / "tests" / "test_old.py").write_text("def test_known_bad():\n    assert False\n")
    (repo / "tests" / "test_bug.py").write_text("import mod\n\ndef test_bug():\n    assert mod.f() == 2\n")
    (repo / "mod.py").write_text("def f():\n    return 1\n")
    assert wf.run_flow(repo, "repro").agent_prompt
    assert wf.run_flow(repo, "repro tests/test_bug.py::test_bug").ok
    assert wf.run_flow(repo, "fix").ok
    (repo / "mod.py").write_text("def f():\n    return 2\n")

    # A baseline taken late: the new tests are already in it, one of them red.
    assert wf.run_flow(repo, "onboard --force").ok
    late = wf.load_state(repo).baseline
    assert late["failing"] == ["tests/test_old.py::test_known_bad"]

    result = wf.run_flow(repo, "verify")

    assert result.ok, result.message
    assert "Already failing before this change" in result.message
    assert "fewer passing" not in result.message


def test_removed_test_lines_are_flagged_but_do_not_block(tmp_path):
    repo = _mid_issue(tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "base"],
        cwd=repo, check=True,
    )
    (repo / "tests" / "test_bug.py").write_text("import mod\n\ndef test_bug():\n    assert mod.f() == 2\n")
    (repo / "mod.py").write_text("def f():\n    return 1\n")
    assert wf.run_flow(repo, "repro tests/test_bug.py::test_bug").ok
    assert wf.run_flow(repo, "fix").ok
    (repo / "mod.py").write_text("def f():\n    return 2\n")
    # The "fix" also rewrote an existing test.
    (repo / "tests" / "test_ok.py").write_text("def test_ok():\n    pass\n")

    result = wf.run_flow(repo, "verify")

    assert result.ok
    assert "lines were removed from existing tests (tests/test_ok.py: -1)" in result.message


def _monorepo(tmp_path):
    """A repository with two independent projects, `svc/a` and `svc/b`."""
    def git(*args):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    for name in ("a", "b"):
        project = tmp_path / "svc" / name
        (project / "tests").mkdir(parents=True)
        (project / "calc.py").write_text("def add(x, y):\n    return x + y\n")
        (project / "tests" / "test_calc.py").write_text(
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n"
            "from calc import add\n\n"
            "def test_add():\n    assert add(2, 3) == 5\n"
        )
        (project / "issues").mkdir()
        (project / "issues" / "001-wrong.md").write_text(f"# 001: {name} adds wrong\n")
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-qm", "init")
    return tmp_path


def test_fingerprint_follows_file_contents_when_the_project_is_a_subfolder(tmp_path):
    # git reports `svc/a/calc.py`, which does not exist relative to `svc/a`.
    # Every edit then hashed the same, so a review of an older diff counted.
    repo = _monorepo(tmp_path)
    a = repo / "svc" / "a"
    (a / "calc.py").write_text("one\n")
    first = wf.diff_fingerprint(a)
    (a / "calc.py").write_text("two\n")
    assert wf.diff_fingerprint(a) != first


def test_scope_ignores_changes_elsewhere_in_the_repository(tmp_path):
    repo = _monorepo(tmp_path)
    a, b = repo / "svc" / "a", repo / "svc" / "b"
    (a / "calc.py").write_text("mine\n")
    before, diff = wf.diff_fingerprint(a), wf.working_diff(a)
    (b / "calc.py").write_text("someone else's work\n")
    (b / "new.py").write_text("untracked\n")
    assert wf.diff_fingerprint(a) == before
    assert wf.working_diff(a) == diff
    assert "someone else" not in diff and "mine" in diff


def test_flow_scope_points_the_workflow_at_one_directory(tmp_path):
    repo = _monorepo(tmp_path)

    picked = wf.run_flow(repo, "scope svc/a")
    assert picked.ok and "Scope set to `svc/a`" in picked.message
    assert wf.project_root(repo) == repo / "svc" / "a"

    onboard = wf.run_flow(repo, "onboard")
    assert onboard.ok and "1 passed" in onboard.message
    # The model is told where it may work, and it is not the whole repository.
    assert "limited to the directory `svc/a`" in onboard.agent_prompt

    assert "a adds wrong" in wf.run_flow(repo, "issue").message
    assert "b adds wrong" not in wf.run_flow(repo, "issue").message
    assert "- Scope: `svc/a`" in wf.run_flow(repo, "status").message


def test_each_directory_keeps_its_own_progress(tmp_path):
    repo = _monorepo(tmp_path)
    wf.run_flow(repo, "scope svc/a")
    wf.run_flow(repo, "onboard")
    wf.run_flow(repo, "issue issues/001-wrong.md")

    wf.run_flow(repo, "scope svc/b")
    fresh = wf.run_flow(repo, "status")
    assert "You are in: onboard" in fresh.message and "adds wrong" not in fresh.message

    back = wf.run_flow(repo, "scope svc/a")
    assert "a adds wrong" in back.message

    listing = wf.run_flow(repo, "scope").message
    assert "`/flow scope svc/a`" in listing and "a adds wrong" in listing
    assert "`/flow scope svc/b`" not in listing, "nothing was recorded there"

    # Started inside a directory: that directory is the scope, same progress.
    assert "a adds wrong" in wf.run_flow(repo / "svc" / "a", "status").message
    assert "You are in: onboard" in wf.run_flow(repo / "svc" / "b", "status").message


def test_scope_can_be_cleared_and_rejects_what_is_not_a_directory(tmp_path):
    repo = _monorepo(tmp_path)
    assert not wf.run_flow(repo, "scope svc/nope").ok
    assert not wf.run_flow(repo, "scope ../outside").ok
    wf.run_flow(repo, "scope svc/a")
    cleared = wf.run_flow(repo, "scope --clear")
    assert cleared.ok and wf.project_root(repo) == repo
    assert "- Scope:" not in wf.run_flow(repo, "status").message


def test_pr_draft_is_shown_and_becomes_the_next_step(tmp_path):
    # The draft lives in .git; a path alone left the user asking where the
    # draft was. And after drafting, "next" still said /flow fix because the
    # review had findings, contradicting the message.
    repo = _monorepo(tmp_path)
    a = repo / "svc" / "a"
    state = wf.FlowState(
        test_command="true", base_branch="main", issue="# 001: a adds wrong", issue_title="001: a adds wrong",
        baseline={"summary": "1 passed"}, repro_target="tests/test_calc.py::test_add",
        repro_red={"summary": "1 failed"}, fix_requested=True, verified={"summary": "2 passed"},
        reviewed_by="reviewer", review_findings="1. Something minor.",
    )
    (a / "calc.py").write_text("def add(x, y):\n    return y + x\n")
    state.verified_diff = state.reviewed_diff = wf.diff_fingerprint(a)
    wf.save_state(a, state)
    assert wf.next_step(wf.load_state(a), a)[0] == "/flow fix"

    draft = wf.run_flow(a, "pr")
    assert draft.ok
    # The draft is a file in the project, not buried in .git, and the model
    # is asked to write the explanation rather than leaving TODOs to the user.
    body = a / "PR-DRAFT.md"
    assert "`svc/a/PR-DRAFT.md`" in draft.message and body.exists()
    assert "## Root cause" in body.read_text()
    assert "that starts with `_TODO:` with real content" in draft.agent_prompt
    assert "return y + x" in draft.agent_prompt, "the model is given the diff"
    assert "Something minor" in draft.agent_prompt, "unaddressed findings go under Not verified"
    # The draft is not part of the change: not in the diff, not in git status.
    assert "PR-DRAFT" not in wf.working_diff(a)
    assert wf.diff_fingerprint(a) == wf.load_state(a).verified_diff
    status = subprocess.run(["git", "status", "--short"], cwd=a, capture_output=True, text=True).stdout
    assert "PR-DRAFT" not in status
    assert not wf.run_flow(a, "pr --create").ok, "unfinished description blocks the pull request"

    # The files and their line counts are listed, in the message and the draft.
    assert "calc.py          +1  -1" in draft.message and "| `calc.py`" not in draft.message
    assert "| `calc.py` | +1 | -1 |" in body.read_text()
    assert "nothing about the workflow tool" in draft.agent_prompt

    # Wording already written survives a second `/flow pr`; facts are refreshed.
    text = body.read_text()
    for todo, written in zip(wf._PROSE.values(), ("Sums came out wrong.", "Operands were swapped.", "Swapped them back.", "Nothing else checked.")):
        text = text.replace(todo, written)
    body.write_text(text)
    again = wf.run_flow(a, "pr")
    assert again.agent_prompt is None
    # ...and the description is printed, not merely said to exist.
    for written in ("Sums came out wrong.", "Operands were swapped.", "Swapped them back.", "Nothing else checked."):
        assert written in again.message
    assert again.message.index("**Problem**") < again.message.index("**Root cause**") < again.message.index("**Fix**")
    assert "Issue as filed" in body.read_text() and "a adds wrong" in body.read_text()
    assert "Full suite after: 2 passed" in again.message
    assert "Operands were swapped." in body.read_text() and "_TODO:" not in body.read_text()

    assert wf.next_step(wf.load_state(a), a)[0] == "/flow pr --create"

    # Having drafted the PR, moving on to another issue needs no --force.
    (a / "issues" / "002-other.md").write_text("# 002: other\n")
    switched = wf.run_flow(a, "issue issues/002-other.md")
    assert switched.ok and "002: other" in switched.message


def test_pr_diff_lists_files_with_line_counts_and_shows_each_diff(tmp_path):
    repo = _monorepo(tmp_path)
    a = repo / "svc" / "a"
    (a / "calc.py").write_text("def add(x, y):\n    return y + x\n")
    (a / "extra.py").write_text("one\ntwo\n")
    (repo / "svc" / "b" / "calc.py").write_text("not mine\n")

    shown = wf.run_flow(a, "pr --diff")
    assert shown.ok and shown.agent_prompt is None
    # Plain aligned text for the terminal, not table markup.
    assert "calc.py           +1  -1" in shown.message
    assert "extra.py (new)    +2  -0" in shown.message
    assert "2 files changed   +3  -1" in shown.message
    assert "| ---" not in shown.message
    assert "### `calc.py`" in shown.message and "+    return y + x" in shown.message
    assert "### `extra.py`" in shown.message and "+two" in shown.message
    assert "not mine" not in shown.message, "other directories are not part of this change"
    assert not (a / "PR-DRAFT.md").exists(), "--diff only looks"


def test_pr_after_the_pull_request_is_open_changes_nothing(tmp_path):
    # Re-running `/flow pr` used to write a fresh, empty draft over an
    # opened pull request's finished description.
    repo = _monorepo(tmp_path)
    a = repo / "svc" / "a"
    wf.save_state(a, wf.FlowState(
        issue="x", issue_title="x", baseline={"summary": "1 passed"},
        pr_url="https://example.test/pull/1",
    ))
    again = wf.run_flow(a, "pr")
    assert again.ok and "already open" in again.message and "https://example.test/pull/1" in again.message
    assert again.agent_prompt is None and not (a / "PR-DRAFT.md").exists()
