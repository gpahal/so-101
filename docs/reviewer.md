# The reviewer: the web page of `review` and `review --runs`

Status: built (`web/review.py`, `web/review.html`, on `web/server.py`), and shown in a browser only on the
tests' recordings and copies of them. How to use it is in the README, under
[Reviewing demos](../README.md#reviewing-demos) and [Reviewing runs](../README.md#reviewing-runs).

`so101 review` (demos) and `so101 review --runs` (`serve` and `run` attempts) each open the same web page,
showing only their kind, and take every decision on it. `--terminal` asks at text prompts instead, without
a viewer, for use over SSH. `record` and `teleop --cameras` show their cameras in Rerun.

## Where it's reached

The page server is the control page's (`web/server.py`): the same never-public checks (Host, Origin,
framing, Funnel and Serve rules, the route watch) and the same bind rule, `[control] bind`: over Tailscale
when it's running, else this Mac only. Its own port, `[control] review_port` (8766), lets a review run while
`serve` does. The terminal prints the address and progress; Ctrl+C stops.

## How it works

- **What's left to review:** demos not yet in `edits.toml`, or attempts with no outcome in either
  `labels.toml` or the run's live outcome events (`review_runs.reviewed`), oldest first.
- **Decisions** go through `review.save` (`edits.toml`) or `review_runs.apply` (the run's `labels.toml`), the
  functions the terminal prompts use, so both ways write identical files; the recordings are never changed.
  `data.save_edits` drops control characters and parses its text before it replaces the file.
- **Frames** are decoded only through `dataset_io.episode_frames`, `FRAMES_PER_REQUEST` (150, 5 s) at a time,
  as JPEGs `WIDTH` (480) px wide, and the server keeps the last `KEEP_BYTES` (80 MB). A frames reply gets
  `SLOWEST_LINK_BYTES_PER_S` (1 Mbit/s, so 32 s for a 4 MB chunk), where any other reply has the connection's
  10 s, and the page asks again twice for one that fails.
- **The timeline** of a run attempt draws the phases as coloured bands (policy, paused, teleop, rewinding,
  and in grey the scripted moves: starting, handover, returning), each interruption, the suspect frames,
  playbacks, `grasp` events, `skill` switches and supervisor decisions; a demo's, the suggested trim and the
  kept range; either's, who drove (`control.driver`) and, in a skills run, the skill (`control.skill`).
- **The page takes a decision** only once the item shown has every camera's first frame on screen (or a
  camera has failed to load), with no decision's POST on its way and no item being fetched. A held key counts
  once, so a second Enter can't decide on an item nobody has seen. A reply for an item the page no longer
  shows only updates the list.

## Server endpoints

`GET /`, `GET /items`, `GET /item/<id>`, `GET /frames/<id>/<camera>?from=&to=` and `POST /decision`, each
described in `web/review.py`'s docstring. A decision names its item, so two tabs decide independently: 404
for an item the review doesn't cover, 400 for a body it doesn't take. An unreadable `edits.toml` or
`labels.toml` is a 409 with the file's error; an unreadable recording a 500, as is any other error a route
raises before replying (`web/server.py`'s `Handler._route`), with the error printed in the terminal.
`review --runs` leaves out an attempt whose episode isn't in the run's episode index, as after a kill during
its save, and names it in the terminal and in `/items`' `notes`. Errors close the connection.

## Tests

`tests/test_server.py` covers the shared server. `tests/test_reviewer.py` makes each decision on the page
and at the prompt and compares the files, and covers the never-public checks on the reviewer's port, every
body `review.html` sends, bodies of the wrong shape, files broken mid-review, a run killed during its second
save, chunked and cut-short videos, and a slow frames body. The script's keys, the wait for the first frames
and its retries need a browser.
