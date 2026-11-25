"""Linear as an issue source for `/flow`.

The real MCP client talks to a real MCP server over stdio: a small stand-in
for Linear's hosted server that implements the three tools `/flow` uses with
the same names, parameters and result shape (read off the live server), and
keeps its issues in a JSON file so the tests can see what was changed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bach.ops import linear, workflow as wf  # noqa: E402

SERVER = '''
import json, sys
from mcp.server.fastmcp import FastMCP

DB = sys.argv[1]
mcp = FastMCP("linear-stand-in")

def load():
    return json.load(open(DB))

@mcp.tool()
def list_issues(limit: int = 50, assignee: str = "", project: str = "",
                fields: list[str] = [], orderBy: str = "") -> str:
    issues = load()["issues"]
    if assignee == "me":
        issues = [i for i in issues if i["assignee"] == "Me Myself"]
    if project:
        issues = [i for i in issues if i["project"] == project]
    return json.dumps({"issues": issues[:limit], "hasNextPage": False})

@mcp.tool()
def save_comment(body: str, issueId: str = "") -> str:
    data = load()
    data.setdefault("comments", []).append({"issueId": issueId, "body": body})
    json.dump(data, open(DB, "w"))
    return json.dumps({"id": "c1"})

@mcp.tool()
def list_projects(limit: int = 50) -> str:
    names = sorted({i["project"] for i in load()["issues"]})
    return json.dumps({"projects": [{"name": n} for n in names], "hasNextPage": False})

@mcp.tool()
def get_issue(id: str) -> str:
    for issue in load()["issues"]:
        if issue["id"] == id:
            return json.dumps(issue)
    raise ValueError(f"Entity not found: Issue - {id}")

@mcp.tool()
def save_issue(id: str, state: str = "") -> str:
    data = load()
    for issue in data["issues"]:
        if issue["id"] == id:
            if state == "started":
                issue["status"], issue["statusType"] = "In Progress", "started"
            json.dump(data, open(DB, "w"))
            return json.dumps(issue)
    raise ValueError(f"Entity not found: Issue - {id}")

mcp.run()
'''


def _row(ident, title, status, status_type, assignee, project="Checkout"):
    return {
        "id": ident, "title": title, "description": f"Details of {title}.",
        "status": status, "statusType": status_type, "assignee": assignee,
        "project": project, "url": f"https://linear.app/acme/issue/{ident}/x",
        "gitBranchName": f"me/{ident.lower()}-{title.lower().replace(' ', '-')}",
    }


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    db = tmp_path / "linear.json"
    db.write_text(json.dumps({"issues": [
        _row("ENG-1", "Totals are wrong", "Todo", "unstarted", "Me Myself"),
        _row("ENG-2", "Export is slow", "In Review", "started", "Me Myself", project="Reports"),
        _row("ENG-3", "Old bug", "Done", "completed", "Me Myself"),
        _row("ENG-4", "Rounding drift", "Backlog", "backlog", "Someone Else"),
    ]}))
    server = tmp_path / "server.py"
    server.write_text(SERVER)
    monkeypatch.setenv("LINEAR_MCP_COMMAND", f"{sys.executable} {server} {db}")
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    return db


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init"],
        cwd=root, check=True,
    )
    assert wf.run_flow(root, "onboard").ok
    return root


def _status(db, ident):
    return next(i["status"] for i in json.loads(db.read_text())["issues"] if i["id"] == ident)


def test_references_are_recognised():
    assert linear.parse_reference("eng-12") == "ENG-12"
    assert linear.parse_reference("https://linear.app/acme/issue/ENG-12/some-title") == "ENG-12"
    assert linear.parse_reference("the total is wrong") is None
    assert linear.parse_reference("#12") is None


def test_linear_lists_my_open_issues_as_commands(tracker, repo):
    result = wf.run_flow(repo, "linear")

    assert result.ok and result.agent_prompt is None
    assert "`/flow issue ENG-1` Totals are wrong (Todo, Checkout)" in result.message
    assert "`/flow issue ENG-2` Export is slow (In Review, Reports)" in result.message
    assert "ENG-3" not in result.message  # done
    assert "ENG-4" not in result.message  # someone else's
    assert _status(tracker, "ENG-1") == "Todo"  # listing changes nothing


def test_linear_project_lists_everyones_open_issues_mine_first(tracker, repo):
    message = wf.run_flow(repo, "linear Checkout").message

    assert message.index("ENG-1") < message.index("ENG-4")
    assert "`/flow issue ENG-1` Totals are wrong (Todo, you)" in message
    assert "`/flow issue ENG-4` Rounding drift (Backlog, Someone Else)" in message
    assert "ENG-2" not in message and "ENG-3" not in message


def test_picking_a_linear_issue_starts_the_flow_and_moves_it_to_in_progress(tracker, repo):
    result = wf.run_flow(repo, "issue ENG-1")

    assert result.ok and result.agent_prompt
    assert "Details of Totals are wrong." in result.agent_prompt
    state = wf.load_state(repo)
    assert state.issue_title == "ENG-1: Totals are wrong"
    assert state.issue_ref == "ENG-1"
    assert state.branch == "me/eng-1-totals-are-wrong"  # Linear's branch name
    assert "ENG-1 moved to In Progress in Linear (was Todo)" in result.message
    assert _status(tracker, "ENG-1") == "In Progress"
    assert "Fixes ENG-1" in wf.pr_body(repo, state)

    # Asking for the same issue again changes nothing, here or in Linear.
    again = wf.run_flow(repo, "issue ENG-1")
    assert not again.ok and "Already working on this issue" in again.message


def test_an_issue_already_under_way_keeps_its_status(tracker, repo):
    result = wf.run_flow(repo, "issue ENG-2")

    assert result.ok
    assert "ENG-2 left as it was in Linear (In Review)" in result.message
    assert _status(tracker, "ENG-2") == "In Review"


def test_unknown_issue_and_missing_key_are_reported_plainly(tracker, repo, monkeypatch):
    missing = wf.run_flow(repo, "issue ENG-999")
    assert not missing.ok and "ENG-999" in missing.message
    assert not wf.load_state(repo).issue

    monkeypatch.delenv("LINEAR_MCP_COMMAND")
    unset = wf.run_flow(repo, "linear")
    assert not unset.ok and "LINEAR_API_KEY" in unset.message


# ---------------------------------------------------------------------------
# Without the `mcp` package (a bare copy on a remote host): plain HTTP
# ---------------------------------------------------------------------------


def _serve_plain_mcp(seen, event_stream):
    """A real HTTP server speaking just enough streamable-HTTP MCP."""
    import http.server
    import json as _json
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((body.get("method"), self.headers.get("Authorization"),
                         self.headers.get("Mcp-Session-Id"), body.get("params")))
            if "id" not in body:
                self.send_response(202)
                self.end_headers()
                return
            if body["method"] == "initialize":
                result = {"protocolVersion": "2025-03-26", "capabilities": {}}
            else:
                listing = {"issues": [
                    {"id": "ENG-7", "title": "Plain", "statusType": "started", "status": "In Progress"},
                    {"id": "ENG-8", "title": "Done", "statusType": "completed"},
                ]}
                result = {"content": [{"type": "text", "text": _json.dumps(listing)}], "isError": False}
            reply = _json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result})
            if event_stream:
                data = f"event: message\ndata: {reply}\n\n".encode()
                kind = "text/event-stream"
            else:
                data, kind = reply.encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Mcp-Session-Id", "sess-1")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize("event_stream", [False, True])
def test_lists_issues_over_plain_http_when_mcp_package_is_missing(monkeypatch, event_stream):
    from bach.ops import linear

    seen = []
    server = _serve_plain_mcp(seen, event_stream)
    try:
        monkeypatch.delenv("LINEAR_MCP_COMMAND", raising=False)
        monkeypatch.setenv("LINEAR_API_KEY", "lin_test")
        monkeypatch.setenv("LINEAR_MCP_URL", f"http://127.0.0.1:{server.server_address[1]}/mcp")
        monkeypatch.setattr(linear, "_have_mcp", lambda: False)
        issues, mine = linear.list_issues()
    finally:
        server.shutdown()

    assert [i.identifier for i in issues] == ["ENG-7"]  # the completed one is not offered
    assert mine == {"ENG-7"}
    assert [m for m, *_ in seen] == ["initialize", "notifications/initialized", "tools/call"]
    assert all(auth == "Bearer lin_test" for _, auth, _, _ in seen)
    assert seen[2][2] == "sess-1"  # the session id from initialize is sent back
    assert seen[2][3]["name"] == "list_issues" and seen[2][3]["arguments"]["assignee"] == "me"


def test_projects_are_listed_to_pick_from_and_never_guessed(tracker, repo):
    # No project given: your issues, then every project as a ready command.
    menu = wf.run_flow(repo, "linear").message
    assert "`/flow linear Checkout`" in menu and "`/flow linear Reports`" in menu

    # Case does not matter, but part of a name is not a match: list instead.
    assert "ENG-1" in wf.run_flow(repo, "linear checkout").message
    wrong = wf.run_flow(repo, "linear check")
    assert not wrong.ok
    assert "no Linear project named `check`" in wrong.message
    assert "`/flow linear Checkout`" in wrong.message and "`/flow linear Reports`" in wrong.message
    assert "ENG-" not in wrong.message


# ---------------------------------------------------------------------------
# A feature: the issue is the spec
# ---------------------------------------------------------------------------


def _describe(db, ident, description):
    data = json.loads(db.read_text())
    next(i for i in data["issues"] if i["id"] == ident)["description"] = description
    db.write_text(json.dumps(data))


SPEC = """Let people download the totals.

## Acceptance criteria
- [ ] `export(rows)` returns CSV text with a header row
1. An empty list gives only the header

## Out of scope
- Excel files
"""


def test_a_feature_is_planned_from_its_spec_instead_of_traced_like_a_bug(tracker, repo):
    _describe(tracker, "ENG-1", SPEC)

    picked = wf.run_flow(repo, "feature ENG-1")

    assert picked.ok
    assert "**Feature:** ENG-1: Totals are wrong" in picked.message
    assert "Acceptance criteria in the spec: 2" in picked.message
    assert _status(tracker, "ENG-1") == "In Progress"
    assert picked.agent_prompt.startswith("Plan this feature against its spec. Do not write")
    assert "1. `export(rows)` returns CSV text with a header row" in picked.agent_prompt
    assert "2. An empty list gives only the header" in picked.agent_prompt
    assert "Excel files" not in picked.agent_prompt.split("Acceptance criteria found")[1].split("Write the plan")[0]
    assert "root cause" not in picked.agent_prompt
    assert wf.load_state(repo).kind == "feature"


def test_the_plan_is_kept_and_posted_to_the_issue_as_a_comment(tracker, repo):
    _describe(tracker, "ENG-1", SPEC)
    wf.run_flow(repo, "feature ENG-1")

    ok, message = wf.record_plan(repo, "1. Where it fits: totals.py\n6. Open questions: none")

    assert ok and "Posted to ENG-1 in Linear as a comment" in message
    assert "**Do next:** `/flow repro`" in message
    comment = json.loads(tracker.read_text())["comments"][0]
    assert comment["issueId"] == "ENG-1" and "Where it fits: totals.py" in comment["body"]
    state = wf.load_state(repo)
    assert state.plan_posted and "Open questions: none" in state.plan
    # The agreed plan travels with the later tasks: the model has no memory.
    assert "Where it fits: totals.py" in wf.run_flow(repo, "repro").agent_prompt


def test_no_tests_are_written_until_the_issue_has_acceptance_criteria(tracker, repo):
    picked = wf.run_flow(repo, "feature ENG-1")  # description has no criteria
    assert picked.ok and "**No acceptance criteria in the spec.**" in picked.message

    blocked = wf.run_flow(repo, "repro")
    assert not blocked.ok and blocked.agent_prompt is None
    assert "the spec has no acceptance criteria" in blocked.message
    assert "ENG-1 in Linear" in blocked.message
    assert not wf.load_state(repo).repro_requested

    # The criteria are added in Linear, not here: the next try re-reads the issue.
    _describe(tracker, "ENG-1", SPEC)
    asked = wf.run_flow(repo, "repro")
    assert asked.ok and "tests from the 2 acceptance criteria" in asked.message
    assert asked.agent_prompt.startswith("Write acceptance tests for this feature. Do not implement it.")
    assert "An empty list gives only the header" in asked.agent_prompt


def test_a_bug_is_unaffected_by_missing_acceptance_criteria(tracker, repo):
    assert wf.run_flow(repo, "issue ENG-1").ok
    asked = wf.run_flow(repo, "repro")
    assert asked.ok and asked.agent_prompt.startswith("Write the smallest regression test")
    assert json.loads(tracker.read_text()).get("comments") is None
    assert wf.record_plan(repo, "a plan")[0] is False


def test_acceptance_criteria_are_read_from_the_usual_ways_of_writing_them():
    find = wf.acceptance_criteria
    assert find("## Acceptance criteria\n- a\n- b\n\n## Notes\n- c") == ["a", "b"]
    assert find("**Acceptance Criteria:**\n\n1. first\n2) second") == ["first", "second"]
    assert find("Acceptance criteria\nThe feature is done when:\n* [x] one\n  more detail\n* two") == ["one", "two"]
    assert find("We should discuss acceptance criteria later.\n- unrelated") == []
    assert find("## Acceptance criteria\n\nTBD") == []
    assert find("no such section\n- a") == []
