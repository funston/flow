<p align="center"><img src="icon.jpeg" width="280" alt="Flow: a black cat at a glowing code panel"></p>

# flow

Slash commands for Claude Code that take a change from issue to pull request
in seven gated steps. The commands run the tests and keep the record; the
agent reads and writes code; you decide at each step.

- Tests are run by the tool, not reported by the agent. Only `/flow-verify`
  can say a change works.
- Each step refuses to run until the one before it is done.
- Progress is saved in the repository's `.git` directory, so `/flow-status`
  can always say where you are and what to type next.
- The review is done by a fresh agent that has not seen the session.

## Install

```
git clone https://github.com/funston/flow.git ~/flow
python3 ~/flow/install.py
```

Start a new Claude Code session and type `/flow-help`.

Needs Python 3.9+, git and Claude Code. The workflow itself uses only the
standard library.

| Optional | For |
| --- | --- |
| GitHub CLI (`gh`) | Opening and reviewing pull requests, GitHub issues |
| `LINEAR_API_KEY` in your shell profile | Listing and picking Linear issues |

The commands are read from the machine the agent runs on. When you work on a
remote host, install there too.

`install.py --dest DIR` writes the commands somewhere else (for example a
project's `.claude/commands`), `--uninstall` removes them. Files it did not
write are never touched.

## A bug

```
/flow-onboard            map the repo, record the test baseline
/flow-linear             list your Linear issues (or /flow-issue to list others)
/flow-issue ENG-142      branch, move the issue to In Progress, trace the cause
/flow-repro              ask for a regression test
/flow-repro <test id>    run it: it has to fail
/flow-fix                smallest correct fix
/flow-verify             regression test passes and nothing else broke
/flow-review             independent review of the diff
/flow-pr                 write the description; --create opens the pull request
```

## A feature

Write the spec in the issue first. It needs a list a test can check:

```
## Acceptance criteria
- export(rows) returns CSV text with a header row
- an empty list gives only the header
```

Then use `/flow-feature` instead of `/flow-issue`:

```
/flow-feature ENG-200    plan against the spec, no code; the plan is posted
                         to the Linear issue as a comment
/flow-repro              acceptance tests, one per criterion
/flow-repro <test id>    they have to fail: the feature does not exist yet
/flow-fix                build the smallest change that satisfies the spec
/flow-verify  /flow-review  /flow-pr      as for a bug
```

`/flow-repro` re-reads the issue and refuses to write tests until it has an
`Acceptance criteria` list. The reviewer checks the diff against the criteria
one by one.

## All commands

`/flow-help` lists them. Besides the steps above: `/flow-status`,
`/flow-scope` (work in one directory of a large repository), `/flow-prs`
(review someone else's pull request), `/flow-reset`, and prompt-only helpers
for steering the agent (`/flow-refocus`, `/flow-prove-it`, `/flow-trim-diff`,
`/flow-restore-test`, `/flow-explain-code`, `/flow-perf`).

## Layout

| Path | What |
| --- | --- |
| `bin/flow` | The command the slash commands call |
| `install.py` | Writes the slash commands into `~/.claude/commands` |
| `commands/` | Prompt-only helpers, installed as they are |
| `src/flow/workflow.py` | The steps, gates and saved state |
| `src/flow/linear.py` | Linear, through its MCP server |
| `src/flow/flow_cli.py` | Command-line front end |
| `tests/` | Tests: real git repositories and real pytest runs |

## Tests

```
python3 -m pip install pytest mcp
python3 -m pytest tests -q
```
