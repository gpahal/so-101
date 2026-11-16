"""`so101-failbench`: benchmark vision models on telling when something has gone wrong in a frame.

Each sample is one instant of a recorded policy attempt on block-container: the top and wrist camera
frames, the moment it was taken at (just after a grasp, just after a release, the end of the attempt, ...)
and a label, `ok` or `wrong` with a category (missed grasp, block beside the jar, ...).
Every model gets the same frames and question and answers whether something is wrong, how sure it is
and which category, and the benchmark scores the answers against the labels.

The models, their adapters and the prompt are the shared judges in `so_101.vision`, which the
supervisor asks too.

Modules:
- `dataset`    `labels.csv` (the curated samples), building the image folder from the recordings, loading it
- `candidates` frames to label from `so101 serve` runs, where the grasp detector would check
- `runner`     calls the models in parallel, caches every answer, keeps to a spending cap
- `metrics`    scores and the Markdown report
- `cli`        the command

The provider SDKs are in the `failbench` dependency group: `uv run --group failbench so101-failbench ...`.
"""
