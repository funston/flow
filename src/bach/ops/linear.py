"""Linear as an issue source for `/flow`, through Linear's MCP server.

`/flow linear [project]` lists issues to pick from, `/flow issue ENG-123`
pulls one in and moves it to In Progress. The MCP tools are called directly
from code rather than through the reasoning model: these are lookups, and
doing them here keeps them fast, free and repeatable, like the rest of the
deterministic half of `/flow`.

Connection:
  * `LINEAR_API_KEY`      personal API key, sent as `Authorization: Bearer`.
  * `LINEAR_MCP_URL`      override the endpoint (default: Linear's hosted server).
  * `LINEAR_MCP_COMMAND`  run a stdio MCP server instead (used by the tests,
                          and by anyone bridging through a local proxy).

Tool names and parameters (`list_issues`, `get_issue`, `save_issue`,
`save_comment`, `list_projects`) were
read off the live server on 2026-11-12; Linear's docs do not list them.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_URL = "https://mcp.linear.app/mcp"
TIMEOUT_SECONDS = 25.0

# ENG-123, or a Linear issue URL containing one.
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]{0,9}-\d+")
_URL = re.compile(r"https://linear\.app/[^/\s]+/issue/([A-Za-z][A-Za-z0-9]{0,9}-\d+)")

# Linear's status types. Anything here is finished and not offered to work on.
_CLOSED = {"completed", "canceled"}
# Types that mean "nobody has started": these get moved to In Progress.
_NOT_STARTED = {"triage", "backlog", "unstarted"}

_FIELDS = [
    "id", "title", "description", "url", "gitBranchName", "status",
    "statusType", "assignee", "project", "team", "priority",
]


class LinearError(RuntimeError):
    """Linear could not be reached or refused the request. The message is
    written for the person at the keyboard."""


@dataclass
class LinearIssue:
    identifier: str
    title: str
    description: str = ""
    status: str = ""
    status_type: str = ""
    url: str = ""
    branch: str = ""
    project: str = ""
    assignee: str = ""

    @property
    def open(self) -> bool:
        return self.status_type not in _CLOSED


def configured() -> bool:
    return bool(os.environ.get("LINEAR_API_KEY") or os.environ.get("LINEAR_MCP_COMMAND"))


def parse_reference(text: str) -> Optional[str]:
    """`ENG-123` or a Linear issue URL -> `ENG-123`; anything else -> None."""
    text = text.strip()
    url = _URL.match(text)
    if url:
        return url.group(1).upper()
    if _IDENTIFIER.fullmatch(text):
        return text.upper()
    return None


# ---------------------------------------------------------------------------
# MCP transport
# ---------------------------------------------------------------------------


async def _call_async(tool: str, arguments: dict) -> Any:
    from mcp import ClientSession

    async def run(read, write) -> Any:
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
            text = "\n".join(
                block.text for block in result.content if getattr(block, "type", "") == "text"
            )
            if result.isError:
                raise LinearError(f"Linear rejected `{tool}`: {text.strip()[:300] or 'no detail'}")
            try:
                return json.loads(text)
            except ValueError:
                return {"text": text}

    command = os.environ.get("LINEAR_MCP_COMMAND")
    if command:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        argv = shlex.split(command)
        params = StdioServerParameters(command=argv[0], args=argv[1:], env=dict(os.environ))
        async with stdio_client(params) as (read, write):
            return await run(read, write)

    key = os.environ.get("LINEAR_API_KEY")
    if not key:
        raise LinearError(
            "LINEAR_API_KEY is not set in the environment Bach is running in. "
            "Export it in your shell profile, then restart the editor or "
            "terminal Bach was started from."
        )
    from mcp.client.streamable_http import streamablehttp_client

    url = os.environ.get("LINEAR_MCP_URL", DEFAULT_URL)
    async with streamablehttp_client(url, headers={"Authorization": f"Bearer {key}"}) as (
        read, write, _,
    ):
        return await run(read, write)


def _leaf(exc: BaseException) -> BaseException:
    """The first real error inside nested ExceptionGroups."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return exc


def _have_mcp() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


def _http_messages(url: str, key: str, payload: dict, session: str) -> tuple[list, str]:
    """POST one JSON-RPC message to a streamable-HTTP MCP server using only
    the standard library. Returns the JSON-RPC messages in the reply (the
    body is either plain JSON or an event stream) and the session id."""
    import urllib.error
    import urllib.request

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "bach-flow",
    }
    if session:
        headers["Mcp-Session-Id"] = session
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as reply:
            session = reply.headers.get("Mcp-Session-Id") or session
            kind = reply.headers.get("Content-Type", "")
            body = reply.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise LinearError("Linear refused the API key (401). Check LINEAR_API_KEY.") from None
        raise LinearError(f"Could not reach Linear: HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise LinearError(f"Could not reach Linear: {exc}") from None
    if not body.strip():
        return [], session
    if "text/event-stream" in kind:
        chunks = [line[5:].strip() for line in body.splitlines() if line.startswith("data:")]
    else:
        chunks = [body]
    messages = []
    for chunk in chunks:
        try:
            messages.append(json.loads(chunk))
        except ValueError:
            continue
    return messages, session


def _call_plain(tool: str, arguments: dict) -> Any:
    """`_call` without the `mcp` package: the same three-message exchange
    (initialize, initialized, tools/call) over plain HTTP. Used where only
    the standard library is available, such as a bare copy on a remote host."""
    key = os.environ.get("LINEAR_API_KEY")
    if not key:
        raise LinearError(
            "LINEAR_API_KEY is not set in the environment this is running in. "
            "Export it in the shell profile on this machine, then restart the "
            "editor or terminal."
        )
    url = os.environ.get("LINEAR_MCP_URL", DEFAULT_URL)
    _, session = _http_messages(url, key, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": "bach-flow", "version": "1"},
        },
    }, "")
    _http_messages(url, key, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
    messages, _ = _http_messages(url, key, {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }, session)
    answer = next((m for m in messages if isinstance(m, dict) and m.get("id") == 2), None)
    if answer is None:
        raise LinearError(f"Linear gave no answer to `{tool}`.")
    if answer.get("error"):
        raise LinearError(f"Linear rejected `{tool}`: {str(answer['error'].get('message', ''))[:300]}")
    result = answer.get("result") or {}
    text = "\n".join(
        block.get("text", "") for block in result.get("content", []) if block.get("type") == "text"
    )
    if result.get("isError"):
        raise LinearError(f"Linear rejected `{tool}`: {text.strip()[:300] or 'no detail'}")
    try:
        return json.loads(text)
    except ValueError:
        return {"text": text}


def _call(tool: str, arguments: dict) -> Any:
    """Call one Linear MCP tool and return its parsed result. Blocking; must
    not be called from inside a running event loop (surfaces run `/flow` in
    a worker thread or from synchronous code)."""
    if not os.environ.get("LINEAR_MCP_COMMAND") and not _have_mcp():
        return _call_plain(tool, arguments)
    try:
        return asyncio.run(asyncio.wait_for(_call_async(tool, arguments), TIMEOUT_SECONDS))
    except LinearError:
        raise
    except asyncio.TimeoutError:
        raise LinearError(f"Linear did not answer within {TIMEOUT_SECONDS:.0f}s.") from None
    except BaseException as exc:  # noqa: BLE001 - anyio wraps errors in groups
        leaf = _leaf(exc)
        if isinstance(leaf, LinearError):
            raise leaf from None
        if isinstance(leaf, (KeyboardInterrupt, SystemExit)):
            raise
        detail = str(leaf)
        if "401" in detail:
            raise LinearError("Linear refused the API key (401). Check LINEAR_API_KEY.") from None
        raise LinearError(f"Could not reach Linear: {type(leaf).__name__}: {detail[:200]}") from None


def _issue(raw: dict) -> LinearIssue:
    return LinearIssue(
        identifier=str(raw.get("id") or ""),
        title=str(raw.get("title") or ""),
        description=str(raw.get("description") or ""),
        status=str(raw.get("status") or ""),
        status_type=str(raw.get("statusType") or ""),
        url=str(raw.get("url") or ""),
        branch=str(raw.get("gitBranchName") or ""),
        project=str(raw.get("project") or ""),
        assignee=str(raw.get("assignee") or ""),
    )


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def list_issues(project: str = "", *, limit: int = 50) -> tuple[list[LinearIssue], set[str]]:
    """Open issues to choose from, plus the identifiers assigned to the caller.

    With no PROJECT: the caller's own open issues. With a PROJECT (name, id
    or slug): every open issue in it, the caller's first.
    """
    base = {"limit": limit, "fields": _FIELDS, "orderBy": "updatedAt"}
    if project:
        project = resolve_project(project)
    mine_args = {**base, "assignee": "me"}
    if project:
        mine_args["project"] = project
    mine = [i for i in map(_issue, _call("list_issues", mine_args).get("issues", [])) if i.open]
    mine_ids = {i.identifier for i in mine}
    if not project:
        return mine, mine_ids
    everyone = _call("list_issues", {**base, "project": project}).get("issues", [])
    others = [i for i in map(_issue, everyone) if i.open and i.identifier not in mine_ids]
    return mine + others, mine_ids


def list_projects() -> list[str]:
    """Names of the Linear projects the caller can see."""
    # Linear refuses a page size above 50, so follow its cursor instead.
    names: list[str] = []
    cursor = ""
    for _ in range(10):
        args: dict = {"limit": 50}
        if cursor:
            args["cursor"] = cursor
        page = _call("list_projects", args)
        names += [str(p.get("name")) for p in page.get("projects", []) if p.get("name")]
        cursor = str(page.get("cursor") or page.get("nextCursor") or "")
        if not page.get("hasNextPage") or not cursor:
            break
    return names


def resolve_project(name: str) -> str:
    """The Linear project called NAME, ignoring case. Linear only accepts an
    exact name; anything else is an error that lists the projects to pick
    from, rather than a guess."""
    names = list_projects()
    for known in names:
        if known.lower() == name.strip().lower():
            return known
    choices = "\n".join(f"- `/flow linear {n}`" for n in names) or "- (no projects)"
    raise LinearError(f"there is no Linear project named `{name}`. Pick one:\n\n{choices}")


def get_issue(identifier: str) -> LinearIssue:
    raw = _call("get_issue", {"id": identifier})
    issue = _issue(raw)
    if not issue.identifier or not issue.title:
        raise LinearError(f"Linear has no issue {identifier}.")
    return issue


def start_issue(issue: LinearIssue) -> str:
    """Move ISSUE to In Progress if nobody has started it. Returns a one-line
    account of what happened, for the workflow log."""
    if issue.status_type not in _NOT_STARTED:
        return f"{issue.identifier} left as it was in Linear ({issue.status or 'unknown status'})"
    raw = _call("save_issue", {"id": issue.identifier, "state": "started"})
    status = str(raw.get("status") or "In Progress") if isinstance(raw, dict) else "In Progress"
    return f"{issue.identifier} moved to {status} in Linear (was {issue.status or 'not started'})"


def add_comment(identifier: str, body: str) -> None:
    """Post BODY (Markdown) as a comment on the issue."""
    _call("save_comment", {"issueId": identifier, "body": body})


__all__ = [
    "LinearError",
    "LinearIssue",
    "add_comment",
    "configured",
    "get_issue",
    "list_issues",
    "list_projects",
    "parse_reference",
    "resolve_project",
    "start_issue",
]
