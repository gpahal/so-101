"""`so101-failbench`: benchmark vision models on telling when something has gone wrong in a frame.

Each sample is one instant of a recorded `so101 serve` or `so101 run` attempt on block-container: the top
and wrist camera frames, the moment it was taken at (just after a grasp, just after a release, the end of
the attempt, ...) and a label, `ok` or `wrong` with a category (missed grasp, block beside the jar, ...),
which a person gives in the labels file `candidates` writes. Every model gets the same frames and question
and answers whether something is wrong, how sure it is and which category, and the benchmark scores the
answers against the labels.

The models, their adapters and the prompt are the shared judges in `so_101.vision`, which the
supervisor asks too.

Modules:
- `dataset`    reading a labels file, building the image folder from the runs, loading it
- `candidates` frames to label from the runs, where the supervisor would check, merged into the labels file
- `runner`     calls the models in parallel, caches every answer, keeps to a spending cap, stops on Ctrl+C
- `metrics`    scores, on the samples every model was asked about and at the supervisor's thresholds, and
               the Markdown report
- `cli`        the command

The provider SDKs are in the `failbench` dependency group: `uv run --group failbench so101-failbench ...`.
"""
