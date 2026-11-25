"""`/flow prs`: list, review, approve and merge someone else's pull request.

GitHub is stood in for by a small `gh` program on PATH that answers from a
JSON file and records what it was asked to do. Everything else is real: a
real repository, a real remote holding the pull request's commits under
`refs/pull/7/head`, a real fetch, and real test runs in a real worktree.
"""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bach.ops import workflow as wf  # noqa: E402

FAKE_GH = r"""#!/usr/bin/env python3
import json, os, sys
db_path = os.environ["FAKE_GH_DB"]
db = json.load(open(db_path))
args = sys.argv[1:]
db.setdefault("calls", []).append(args)
def save():
    json.dump(db, open(db_path, "w"))
if args[:2] == ["pr", "list"]:
    save(); print(json.dumps([p for p in db["prs"] if p["state"] == "OPEN"])); sys.exit(0)
pr = next((p for p in db["prs"] if str(p["number"]) == args[2]), None) if len(args) > 2 else None
if pr is None:
    save(); print("no pull requests found", file=sys.stderr); sys.exit(1)
if args[:2] == ["pr", "view"]:
    save(); print(json.dumps(pr)); sys.exit(0)
if args[:2] == ["pr", "review"]:
    if pr["author"]["login"] == "me":
        save(); print("Can not approve your own pull request", file=sys.stderr); sys.exit(1)
    pr["reviewDecision"] = "APPROVED" if "--approve" in args else "CHANGES_REQUESTED"
    save(); sys.exit(0)
if args[:2] == ["pr", "merge"]:
    pr["state"] = "MERGED"; save(); sys.exit(0)
save(); sys.exit(2)
"""


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                   cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A clone whose remote has pull request #7: a coworker's change to calc.py."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    work = tmp_path / "work"
    work.mkdir()
    (work / "tests").mkdir()
    (work / "calc.py").write_text("def add(x, y):\n    return x + y\n")
    (work / "tests" / "test_calc.py").write_text(
        "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n"
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    git(work, "init", "-q", "-b", "main")
    git(work, "add", "-A")
    git(work, "commit", "-qm", "init")
    git(work, "remote", "add", "origin", str(remote))
    git(work, "push", "-q", "origin", "main")
    # The coworker's branch: breaks add(), pushed as the pull request's head.
    git(work, "checkout", "-q", "-b", "coworker")
    (work / "calc.py").write_text("def add(x, y):\n    return x - y\n")
    git(work, "commit", "-qam", "Speed up add")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work, capture_output=True, text=True).stdout.strip()
    git(work, "push", "-q", "origin", "HEAD:refs/pull/7/head")
    git(work, "checkout", "-q", "main")
    git(work, "branch", "-qD", "coworker")

    db = tmp_path / "gh.json"
    db.write_text(json.dumps({"prs": [{
        "number": 7, "title": "Speed up add", "author": {"login": "dana"},
        "headRefName": "coworker", "baseRefName": "main", "headRefOid": head,
        "isDraft": False, "state": "OPEN", "reviewDecision": "", "mergeable": "MERGEABLE",
        "statusCheckRollup": [], "url": "https://example.test/pull/7", "body": "Faster.",
        "additions": 1, "deletions": 1,
        "files": [{"path": "calc.py", "additions": 1, "deletions": 1}],
    }]}))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(FAKE_GH.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GH_DB", str(db))
    return work, db


def calls(db):
    return json.loads(db.read_text())["calls"]


def pr(db):
    return json.loads(db.read_text())["prs"][0]


def test_list_shows_open_pull_requests_as_commands(repo):
    work, _ = repo
    for argline in ("prs", "prs list"):
        listing = wf.run_flow(work, argline)
        assert listing.ok
        assert "`/flow prs 7` Speed up add (by dana, +1 -1, not reviewed)" in listing.message


def test_review_reads_the_pull_request_without_touching_the_working_tree(repo):
    work, db = repo
    review = wf.run_flow(work, "prs 7")
    assert review.ok
    assert "Reviewing pull request #7 Speed up add" in review.message
    assert "calc.py" in review.message and "+1  -1" in review.message
    # The reviewer is given the change and told how to read the PR's files.
    assert "return x - y" in review.agent_prompt and "refs/bach/pr-7" in review.agent_prompt
    assert "do not post anything to GitHub" in review.agent_prompt
    shown = subprocess.run(["git", "show", "refs/bach/pr-7:calc.py"], cwd=work, capture_output=True, text=True)
    assert "return x - y" in shown.stdout
    # Still on main, file untouched, nothing sent to GitHub.
    assert (work / "calc.py").read_text().endswith("return x + y\n")
    assert all(call[:2] != ["pr", "review"] for call in calls(db))
    # "#7" is accepted too.
    assert wf.run_flow(work, "prs #7").ok


def test_review_with_test_runs_the_suite_on_the_pull_requests_code(repo):
    work, _ = repo
    wf.run_flow(work, "onboard")
    review = wf.run_flow(work, "prs 7 test")
    assert "1 failed" in review.message, "the coworker's change breaks add()"
    assert "1 failed" in review.agent_prompt
    worktrees = subprocess.run(["git", "worktree", "list"], cwd=work, capture_output=True, text=True).stdout
    assert len(worktrees.strip().splitlines()) == 1, "the throwaway worktree is gone"
    assert (work / "calc.py").read_text().endswith("return x + y\n")


def test_cannot_answer_a_pull_request_that_was_not_reviewed(repo):
    work, db = repo
    blocked = wf.run_flow(work, "prs 7 approve")
    assert not blocked.ok and "has not been reviewed here" in blocked.message
    assert all(call[:2] != ["pr", "review"] for call in calls(db))


def test_new_commits_after_the_review_need_another_look(repo):
    work, db = repo
    wf.run_flow(work, "prs 7")
    data = json.loads(db.read_text())
    data["prs"][0]["headRefOid"] = "f" * 40
    db.write_text(json.dumps(data))
    blocked = wf.run_flow(work, "prs 7 approve")
    assert not blocked.ok and "new commits since you reviewed it" in blocked.message


def test_request_changes_sends_the_note_as_written(repo):
    work, db = repo
    wf.run_flow(work, "prs 7")
    assert not wf.run_flow(work, "prs 7 changes").ok, "a note is required"
    sent = wf.run_flow(work, "prs 7 changes calc.py:2 subtracts.\nPlease add a test.")
    assert sent.ok and "Changes requested" in sent.message
    assert pr(db)["reviewDecision"] == "CHANGES_REQUESTED"
    assert calls(db)[-1] == ["pr", "review", "7", "--request-changes", "--body", "calc.py:2 subtracts.\nPlease add a test."]


def test_approve_then_merge(repo):
    work, db = repo
    wf.run_flow(work, "prs 7")
    not_yet = wf.run_flow(work, "prs 7 merge")
    assert not not_yet.ok and "has not been approved" in not_yet.message
    assert pr(db)["state"] == "OPEN"

    approved = wf.run_flow(work, "prs 7 approve")
    assert approved.ok and "`/flow prs 7 merge`" in approved.message
    merged = wf.run_flow(work, "prs 7 merge")
    assert merged.ok and "Merged" in merged.message and pr(db)["state"] == "MERGED"
    assert calls(db)[-1] == ["pr", "merge", "7", "--squash"]


def test_failing_checks_block_approval_and_merge(repo):
    work, db = repo
    data = json.loads(db.read_text())
    data["prs"][0]["statusCheckRollup"] = [{"name": "unit", "conclusion": "FAILURE"}]
    data["prs"][0]["reviewDecision"] = "APPROVED"
    db.write_text(json.dumps(data))
    review = wf.run_flow(work, "prs 7")
    assert "FAILING (unit)" in review.message
    assert not wf.run_flow(work, "prs 7 approve").ok
    blocked = wf.run_flow(work, "prs 7 merge")
    assert not blocked.ok and "checks are failing (unit)" in blocked.message
    assert pr(db)["state"] == "OPEN"


def test_github_errors_are_reported_plainly(repo):
    work, db = repo
    data = json.loads(db.read_text())
    data["prs"][0]["author"]["login"] = "me"
    db.write_text(json.dumps(data))
    wf.run_flow(work, "prs 7")
    refused = wf.run_flow(work, "prs 7 approve")
    assert not refused.ok and "Can not approve your own pull request" in refused.message
    missing = wf.run_flow(work, "prs 99")
    assert not missing.ok and "no pull requests found" in missing.message
    assert not wf.run_flow(work, "prs approve").ok
