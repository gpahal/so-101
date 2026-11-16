# The reviewer: a web page for `review` and `review --runs`, in place of Rerun

Status: built (`web/review.py`, `web/review.html`, `web/server.py`), and the README's Reviewing demos and
Reviewing runs sections describe it; run only on the tests' recordings and scratch copies of them in a
browser, not yet on real ones. `record`, `teleop --cameras` and the failure benchmark keep Rerun.

`so101 review` (demos) and `so101 review --runs` (`serve` and `run` attempts) each open the same web page,
showing only their kind, and take every decision on it. `--terminal` keeps today's text prompts, without a
viewer, for use over SSH.

## Where it's reached

The page server is the control page's, made shared (`web/server.py`): the same never-public checks
(Host, Origin, framing, Funnel and Serve rules, the route watch) and the same bind rule, `[control] bind`:
over Tailscale when it's running, else this Mac only. Its own port, `[control] review_port` (8766), so a
review can run while `serve` does. The terminal prints the address and progress; Ctrl+C stops.

## What it shows

- **A list** of what's left to review, oldest first: demos not yet in `edits.toml`, or attempts with no
  outcome in either `labels.toml` or the run's live outcome events. `--all` lists every one;
  `--episode SESSION/N` or `--episode RUN[/N]` opens one.
- **Both cameras**, side by side, from frames the Mac decodes once per episode and sends as JPEGs (480 px
  wide). The page holds them, so scrubbing is instant in any browser, a phone's too. About 25 MB per 15 s
  episode, a few seconds to load; the server keeps the last few episodes decoded.
- **A timeline** in seconds under them, with a play/pause and frame steps, and on it:
  - for a run attempt: the phases as coloured bands (policy, paused, teleop, rewinding, returning), each
    interruption, the suspect frames as a shaded range, playbacks, `grasp` events, `skill` switches and
    supervisor decisions;
  - for a demo: the suggested trim and the kept range.
- **The joints** as plots against the same timeline, the gripper on its own scale.

## What it takes

Buttons, with a key for each:
- **A demo:** accept the suggestion (Enter), keep it all (`k`), keep a range set by dragging the trim
  handles or with `[` and `]` at the playhead, delete with a reason (`d`), skip (`s`).
- **An attempt:** for each interruption, keep, set (drag, or `[` and `]`), clear (`c`) or reset to automatic
  (`a`) its suspect frames; then the outcome (`s`ucceeded, `f`ailed, `i`gnored, `u`nlabelled), Enter to
  confirm.

Each decision is written straight away by the same functions the terminal prompts use, to `edits.toml` or
the run's `labels.toml`, so both ways write identical files. The recordings are never changed.

## Server endpoints

`GET /` (the page), `GET /items` (the list), `GET /item/<id>` (its fps, frame count, joints, phases, events,
interruptions, suggestion and current decision), `GET /frames/<id>/<camera>?from=&to=` (JPEG frames, packed),
`POST /decision` (`{"item", ...}`; refused unless it names the item shown). Errors close the connection, as
the control page's do.

## Tests

The shared server keeps the control page's tests and adds the reviewer's endpoints: a demo and a run
attempt listed and opened, frames returned for the right range, each decision writing the same
`edits.toml` or `labels.toml` as the terminal path, a live-labelled attempt skipped, and the never-public
checks on the reviewer's port.
