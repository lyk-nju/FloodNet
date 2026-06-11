# Runtime Debug Evaluation Refactor Design

## Goal

Runtime evaluation is primarily a debugging tool. It should make it easy to answer which component is responsible for a bad result:

- LDF cannot follow a good root condition.
- RootRefiner predicts the wrong duration or number of tokens.
- RootRefiner predicts bad XZ points.
- RootRefiner predicts bad heading, forward delta, or yaw delta.
- Runtime coordinate transforms or online route edits are inconsistent with the web demo.

The refactor will make the runtime runner web-demo-equivalent, split responsibilities into focused modules, and write artifacts into a directory structure that matches the intended comparisons.

## Terminology

- `gtroot`: the root condition is built from dataset motion 7D extracted from the original motion.
- `rootrefiner`: the root condition is predicted by RootRefiner and then passed to LDF as 7D root condition.
- `notraj`: no trajectory/root condition is passed to LDF.
- `web_stream`: one runtime streaming generation path equivalent to the web demo. It uses `stream_generate_step` and the same RootPlan/7D condition bridge. There is no separate `step_stream` versus `step_generate` semantic split.
- `rotation`: applies a world-space rotation to the root condition before running `web_stream`.
- `turn`: applies an online route edit before running `web_stream`.
- `root_diagnostics`: compares `gtroot` and `rootrefiner` root conditions before LDF generation, using the same experiment transform.

## Output Layout

Runtime outputs will live under `eval/output_eval/runtime`.

```text
eval/output_eval/runtime/
  step_485000/
    20260611_220231/
      manifest.json
      summary.json
      records.csv

      gtroot/
        source.json
        web_stream/
          metrics.json
          video/
          plots/
          samples/
        rotation/
          rot_010/
          rot_020/
          ...
          rot_090/
        turn/
          base/
          delay/
            delay_005/
            delay_010/
            delay_020/
          blend/
            blend_002/
            blend_004/
            blend_008/

      rootrefiner/
        source.json
        web_stream/
        rotation/
        turn/

      notraj/
        source.json
        web_stream/

      root_diagnostics/
        summary.json
        summary.csv
        web_stream/
          gtroot_vs_rootrefiner/
            metrics.json
            root_xz_compare.png
            heading_compare.png
            token_compare.png
            root_plans.npz
        rotation/
          rot_010/
            gtroot_vs_rootrefiner/
          ...
          rot_090/
            gtroot_vs_rootrefiner/
        turn/
          base/
            gtroot_vs_rootrefiner/
          delay/
            delay_020/
              gtroot_vs_rootrefiner/
          blend/
            blend_004/
              gtroot_vs_rootrefiner/
```

Each leaf experiment directory writes its own `metrics.json`, optional rendered video, trajectory plots, and raw NumPy artifacts. The top-level `summary.json` and `records.csv` aggregate all generation metrics. The `root_diagnostics` summary aggregates only pre-LDF root condition comparisons.

## Experiment Defaults

The first implementation will keep the scope small and deterministic:

- Dataset/sample: HumanML3D sample `001168` by default, with CLI support for other sample IDs.
- Root sources: `gtroot`, `rootrefiner`, `notraj`.
- Base generation: `web_stream`.
- Rotation angles: `10,20,30,40,50,60,70,80,90` degrees.
- Turn base: immediate `rot30`.
- Turn delays: `delay_005`, `delay_010`, `delay_020`.
- Turn blends: `blend_002`, `blend_004`, `blend_008`.

Rotation is applied consistently to XZ and heading:

- XZ positions rotate in world space around the chosen anchor.
- Heading `cos/sin` rotates by the same yaw offset.
- `fwd_delta` and `yaw_delta` remain physically consistent and are not treated as world XZ vectors.

## Module Boundaries

The refactor will keep model inference behavior stable while moving responsibilities out of `eval/runtime/benchmark.py`.

```text
eval/runtime/
  benchmark.py      # CLI, config loading, high-level orchestration
  experiments.py    # experiment matrix: root sources, rotation, turn variants
  root_sources.py   # gtroot/rootrefiner/notraj construction
  transforms.py     # 7D rotation and local/world coordinate transforms
  runners.py        # web-demo-equivalent stream_generate_step runner
  artifacts.py      # output layout, manifests, summaries, plots, raw arrays
  metrics.py        # existing metrics plus small additions if needed
```

Existing `cases.py` can remain as a compatibility layer during migration. New debug runs should use `experiments.py`.

## Data Flow

1. `benchmark.py` loads the LDF checkpoint, VAE checkpoint, optional RootRefiner checkpoint, config, and dataset sample.
2. `experiments.py` expands the requested debug matrix into concrete experiment specs.
3. `root_sources.py` builds a root condition for each source:
   - `gtroot`: extract motion 7D from dataset motion.
   - `rootrefiner`: run RootRefiner and convert prediction to LDF 7D root condition.
   - `notraj`: produce an explicit no-condition payload.
4. `transforms.py` applies the experiment transform, such as rotation or turn edit, to the root condition.
5. `runners.py` runs the web-demo-equivalent streaming generation with `stream_generate_step`.
6. `metrics.py` computes generation metrics against the correct target for that experiment.
7. `artifacts.py` writes per-experiment metrics, plots, videos, raw arrays, top-level summaries, and root diagnostics.

## Root Diagnostics

`root_diagnostics` compares `gtroot` and `rootrefiner` before LDF generation. It must use the same transform as the generation experiment. For example, `rotation/rot_090` compares RootRefiner root 7D against GT motion 7D after both are represented in the `rot90` experiment frame.

Metrics include:

- `num_token_gt`
- `num_token_pred`
- `num_token_abs_error`
- `duration_frame_abs_error`
- `xyz_ADE`
- `xyz_FDE`
- `x_AE_mean`
- `z_AE_mean`
- `endpoint_xz_error`
- `heading_mae_deg`
- `yaw_delta_mae`
- `fwd_delta_mae`
- `path_arc_ADE`
- `path_chamfer`
- `cross_track_mean`
- `cross_track_max`

Artifacts include:

- `root_plans.npz`: GT and RootRefiner 7D arrays plus masks/tokens.
- `root_xz_compare.png`: XZ path overlay.
- `heading_compare.png`: heading/yaw overlay.
- `token_compare.png`: duration and token boundary comparison.
- `metrics.json`: structured per-experiment root comparison metrics.

## Error Handling

The runner should fail fast for configuration problems:

- `rootrefiner` source requested without a RootRefiner checkpoint.
- Root 7D arrays have inconsistent dimensions or invalid masks.
- A rotation/turn experiment is requested for `notraj` when no root condition exists.
- The runtime runner produces a local/world coordinate mismatch.
- Required dataset sample or text segment is missing.

Failure messages should include the root source, experiment family, sample ID, and output directory.

## Backward Compatibility

The current `eval/runtime/benchmark.py` CLI should keep working for existing commands while defaulting new debug output to `eval/output_eval/runtime`. Legacy `suite/mode/condition_variant` names can be translated into the new experiment specs during migration. Old result folders under `eval/result` do not need to be rewritten.

## Test Plan

Add focused tests before implementation:

- Output layout builder creates the expected `runtime/step_485000/<run_id>/<root_source>/...` directories.
- Experiment matrix expands root sources, rotation angles, and turn settings correctly.
- `gtroot` and `rootrefiner` diagnostics use the same transformed reference frame.
- 7D rotation preserves array shape and rotates XZ plus heading consistently.
- `notraj` only runs `web_stream` and is excluded from root diagnostics.
- Existing runtime coordinate-transform tests keep passing.

After implementation, run:

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest \
  tests/test_stream_benchmark_rootplan.py \
  tests/test_stream_eval_metrics.py \
  tests/test_runtime_artifacts.py \
  tests/test_runtime_experiments.py -q
```

Then run one smoke runtime evaluation on sample `001168` and inspect:

- `summary.json`
- `records.csv`
- `root_diagnostics/summary.json`
- one `gtroot/web_stream` video
- one `rootrefiner/web_stream` video
- one `rotation/rot_090` plot

## Implementation Sequence

1. Add `artifacts.py` and tests for the new output layout.
2. Add `experiments.py` and tests for the runtime debug matrix.
3. Move 7D transform helpers into `transforms.py` and keep existing transform tests passing.
4. Add `root_sources.py` and normalize source names to `gtroot`, `rootrefiner`, and `notraj`.
5. Extract the web-demo-equivalent streaming loop into `runners.py`.
6. Add `root_diagnostics` metric and plot writing.
7. Slim `benchmark.py` down to CLI orchestration and compatibility translation.
8. Run unit tests, then a sample `001168` runtime smoke test.
