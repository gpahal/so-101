# The reviewer: a web page for `review` and `review --runs`, in place of Rerun

Status: built (`web/review.py`, `web/review.html`, `web/server.py`), and the README's Reviewing demos and
Reviewing runs sections describe it; run only on the tests' recordings and scratch copies of them in a
browser, not yet on real ones. `record` and `teleop --cameras` keep Rerun.

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
- **Both cameras**, side by side, from frames the Mac decodes 5 s at a time, as the page asks for them, and
  sends as JPEGs (480 px wide). The page holds them, so scrubbing is instant in any browser, a phone's too.
  About 25 MB per 15 s episode, a few seconds to load; the server keeps the last 80 MB of JPEGs. A chunk
  may take 32 s to send (1 Mbit/s for about 4 MB), where any other reply has the connection's 10 s, and
  the page asks again twice for one that fails. When a video stops decoding partway, the frames before
  that are sent, and the next request says why.
- **A timeline** in seconds under them, with a play/pause and frame steps, and on it:
  - for a run attempt: the phases as coloured bands (policy, paused, teleop, rewinding, returning), each
    interruption, the suspect frames as a shaded range, playbacks, `grasp` events, `skill` switches and
    supervisor decisions;
  - for a demo: the suggested trim and the kept range;
  - for either: who drove (`control.driver`) and, in a skills run, the skill (`control.skill`), each
    span named.
- **The joints** as plots against the same timeline, the gripper on its own scale.

## What it takes

Buttons, with a key for each:
- **A demo:** accept the suggestion (Enter), keep it all (`k`), keep a range set by dragging the trim
  handles or with `[` and `]` at the playhead, delete with a reason (`d`), skip (`s`).
- **An attempt:** for each interruption, keep, set (drag, or `[` and `]`), clear (`c`) or reset to automatic
  (`a`) its suspect frames; then the outcome (`s`ucceeded, `f`ailed, `i`gnored, `u`nlabelled), Enter to
  confirm.

Each decision is written straight away by the same functions the terminal prompts use, to `edits.toml` or
the run's `labels.toml`, so both ways write identical files. The recordings are never changed. A delete
reason is joined by single spaces, as the terminal's words are, and `data.save_edits` drops control
characters and parses its text before it replaces the file.

The page takes a decision only once the item shown has every camera's first frame on screen (or a camera
has failed to load), with no decision's POST on its way and no item being fetched: from a click in the
list, or a decision that moves on, until the next item's first frames show, Enter and the decision keys do
nothing and the decision buttons are dimmed. A held key counts once, so a second Enter can't decide on an
item nobody has seen. A reply for an item the
page no longer shows only updates the list. A button clicked with the pointer gives up the focus, so
Space and Enter then do what the keys say rather than press it again.

## Server endpoints

`GET /` (the page), `GET /items` (the list), `GET /item/<id>` (its fps, frame count, joints, phases, events,
interruptions, suggestion and current decision), `GET /frames/<id>/<camera>?from=&to=` (JPEG frames, packed),
`POST /decision` (`{"item", ...}`, naming its item: the server keeps no item "shown", so two tabs can each
decide on their own; 404 for an item the review doesn't cover, 400 for a body it doesn't take). An
`edits.toml` or `labels.toml` that can't be read answers any request with 409 and the file's error; a
recording that can't be read, 500, including a video PyAV can't decode and a Parquet file that won't parse.
Any other error a route raises before it has replied is a 500 naming the error (`web/server.py`'s
`Handler._route`), with the error printed in the terminal, so the page never sees a dropped connection. `review --runs` leaves out an attempt whose episode isn't in the run's
episode index, as after a kill during its save, and says so in the terminal and in `/items`' `notes`.
Errors close the connection, as the control page's do. The full list is in `web/review.py`'s docstring.

## Tests

The shared server keeps the control page's tests and adds the reviewer's endpoints: a demo and a run
attempt listed and opened, frames returned for the right range, each decision writing the same
`edits.toml` or `labels.toml` as the terminal path, a live-labelled attempt skipped, and the never-public
checks on the reviewer's port. Also: every body the page's script sends (each call's text is checked
against `review.html`), a decision for an item other than the one opened last, bodies of the wrong shape,
a `labels.toml` and an `edits.toml` broken mid-review, a run hard-killed during its second save, frames
decoded per chunk, a video cut short, a frames body sent slower than the connection's timeout, the
driver bands' names, and `save_edits` with control characters. The script's own behaviour (keys, the wait for
the first frames, retries) needs a browser.
