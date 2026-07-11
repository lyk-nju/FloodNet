# Dataset NoiseInitializer Training Design

## Goal

Promote the single-sample residual NoiseInitializer prototype into a formal
dataset-level Stage 1 trainer. The pilot trains on the first 300 samples from
HumanML3D `train_difficult.txt`, validates on the first 50 samples from
`val.txt`, dynamically resamples Gaussian roll-in noise, and uses a bounded
replay buffer instead of a permanent static snapshot pool.

The formal entry point is `train_initializer.py`. The existing
`tools/train_noise_initializer.py` remains a single-sample overfit and debug
entry point.

## Scope

The first dataset-level version includes:

- Gaussian roll-in Stage 1 training only.
- Dynamic Gaussian seeds for every collected sample roll-in.
- Two to four randomly selected valid decision commits per roll-in.
- Immutable original Gaussian base noise for every frontier assignment.
- Effect-aligned differentiable rollout horizons.
- Raw residual-ratio regularization plus applied residual hard clipping.
- A finite FIFO replay buffer with weighted commit sampling.
- Fixed held-out sample and seed validation.
- Resumable latest and best checkpoints.

It does not include initializer-rollin Stage 2, Web runtime integration,
token-level text attention, stochastic `(mu, sigma)` prediction, or additional
body-quality losses. Those remain follow-up work after held-out motion gains are
established.

## File Structure

```text
train_initializer.py
configs/noise_initializer_train.yaml

models/noise_initializer.py

utils/training/noise_initializer/
    contracts.py
    dataset_builder.py
    replay_buffer.py
    collector.py
    objectives.py
    trainer.py
    validation.py
    checkpointing.py

    replay_runner.py
    overfit_runner.py
```

Responsibilities:

- `train_initializer.py`: configuration, frozen runtime construction, trainer
  construction, resume, and top-level failure reporting.
- `contracts.py`: immutable dataset replay snapshot and base-noise assignment
  contracts.
- `dataset_builder.py`: HumanML3D split construction, deterministic sample
  limiting, and sample metadata normalization.
- `replay_buffer.py`: bounded FIFO storage, weighted sampling, and coverage
  diagnostics.
- `collector.py`: dynamic seed generation, Gaussian stream roll-in, commit
  selection, and snapshot collection.
- `objectives.py`: effect-aligned horizon resolution and residual-ratio
  objectives.
- `trainer.py`: alternating collection and optimizer updates, logging,
  validation scheduling, and checkpoint scheduling.
- `validation.py`: fixed held-out full rollout evaluation and aggregate metrics.
- `checkpointing.py`: latest/best state, optimizer and RNG state, and replay
  metadata persistence.
- `replay_runner.py` and `overfit_runner.py`: existing single-sample diagnostic
  paths; they remain compatible but do not own formal dataset training.

## Data Contract

Each replay item is identified by:

```text
(sample_id, caption_id, gaussian_seed, absolute_commit_index)
```

It stores:

- the complete restorable LDF stream state;
- the complete restorable VAE stream cache;
- committed history and active noisy-state context;
- strict zero-update frontier token IDs;
- immutable original Gaussian frontier noise;
- currently assigned frontier noise and assignment version;
- runtime trajectory payload and masks;
- committed latent prefix;
- absolute target trajectory and validity mask;
- generated world anchor;
- valid and requested affected-frame counts.

The immutable source rule is:

```text
assigned_frontier_zT = original_base_zT + alpha * clipped_delta_zT
```

Recomputing a frontier assignment replaces `assigned_frontier_zT`. It never
uses a previously assigned value as the next base. The strict frontier remains
restricted to tokens with `token_update_count == 0`.

## Dataset Construction

The pilot configuration uses:

```text
train: HumanML3D/train_difficult.txt, first 300 samples
validation: HumanML3D/val.txt, first 50 samples
```

Sample limiting is deterministic and preserves split-file order. Dataset
construction reuses the existing HumanML3D configuration and normalization
contracts used by LDF training. Caption IDs and sample IDs are retained in all
snapshot and validation records.

Training seeds are generated dynamically from a checkpointed RNG. Validation
uses a fixed configured seed list that does not overlap the training seed
stream.

## Collection Flow

One collection operation performs:

1. Select one training sample.
2. Draw a fresh Gaussian seed from the collector RNG.
3. Initialize the complete rolling latent cache from that seed.
4. Perform a Gaussian-only stream roll-in for the sample.
5. Enumerate decision commits at `optimize_every_tokens` intervals.
6. Filter commits that lack a strict frontier or enough valid affected frames.
7. Uniformly choose between two and four valid commits without replacement.
8. Capture immutable snapshots and add them to the replay buffer.

The collector never applies the learned initializer in this Stage 1 design.
Consequently, every collected history belongs to the Gaussian roll-in policy,
while its seed and selected commits change across collection cycles.

## Replay Buffer

The buffer has a fixed capacity, defaulting to 10,000 snapshots. New snapshots
are appended; the oldest snapshots are evicted when capacity is exceeded.

Sampling weight is:

```text
valid_frame_ratio * commit_progress_weight
```

Late commits at or beyond 55 percent relative progress receive a default
multiplier of 3.0. Truncated tail snapshots are downweighted by
`valid_affected_frames / requested_affected_frames`.

The buffer reports:

- current size and eviction count;
- unique sample, caption, seed, and commit coverage;
- early/late commit sampling counts;
- valid-frame ratio statistics.

Replay tensor payloads are stored on CPU and moved to the training device only
for the sampled optimizer step. This prevents buffer capacity from scaling GPU
memory use.

## Training Objective

For a sampled snapshot, the initializer predicts raw residual noise from
history, active state, text, trajectory, frontier offsets, and immutable base
noise.

The rollout length is effect aligned:

```text
no_effect_tokens = first_frontier_offset
rollout_tokens = no_effect_tokens + desired_affected_tokens
```

The pilot defaults to ten desired affected tokens. Target windows and masks use
the same token-to-frame mapping as the rollout.

Residual control uses two separate mechanisms:

```text
raw_ratio = ||raw_delta_zT|| / ||original_base_zT||
applied_delta = clip_to_ratio(raw_delta_zT, max_delta_norm_ratio)
```

The loss is:

```text
L = L_root_xz
  + lambda_vel * L_root_velocity
  + lambda_raw_ratio * raw_ratio^2
```

Applied hard clipping remains the runtime safety boundary. The regularizer is
computed before clipping so it continues to provide radial gradients after the
hard boundary is reached. Applied ratio and clip saturation remain diagnostics,
not the only residual-size objective.

The frozen LDF and VAE participate in differentiable execution but never
receive parameter gradients. Committed history, active state, and targets are
detached context.

## Alternating Trainer

The default cycle is:

```text
collect 8 sample roll-ins
train 32 replay optimizer steps
repeat
```

Collection and optimization execute in one process in the first version. This
keeps frozen model ownership and VAE cache semantics explicit. The interface
allows collection to move to a separate worker in a later version without
changing replay contracts.

Training logs every 50 optimizer steps and records:

- total, trajectory, velocity, and raw-ratio losses;
- raw and applied residual ratios;
- clip saturation;
- gradient norm;
- sampled sample, seed, and commit;
- replay size and coverage;
- collection and optimization wall time.

## Validation

Validation uses the first 50 `val.txt` samples and fixed held-out Gaussian
seeds. It compares:

```text
Gaussian baseline
initializer alpha=0
initializer configured alpha
```

Required aggregate metrics are:

- full rollout ADE, FDE, MSE, heading error, and path ratio;
- affected-window ADE/FDE/MSE;
- sample-level and window-level harm rates;
- alpha-zero maximum and mean difference from Gaussian;
- clip saturation and applied residual ratio.

`alpha=0` must match Gaussian numerically. Best checkpoint selection uses
validation mean ADE subject to alpha-zero equivalence and finite metric checks.
Validation outputs preserve per-sample metrics so aggregate improvements cannot
hide a high harm rate.

## Checkpointing And Resume

Every checkpoint interval saves:

- initializer state;
- optimizer state;
- global optimizer and collection steps;
- Python and Torch RNG states, including collector RNG;
- active configuration;
- replay metadata and serialized replay items;
- best validation metric and validation history.

`latest.pt` is replaced atomically. `best.pt` is updated only after a valid
held-out evaluation. Resume restores the same seed stream and replay sampling
sequence rather than merely reloading model weights.

## Error Handling

- Invalid or too-short samples are skipped with structured counters.
- A collection cycle that produces no valid snapshots raises an error after
  reporting sample IDs and rejection reasons.
- Non-finite loss or gradients prevent the optimizer update and save a failure
  checkpoint containing the sampled snapshot identity.
- Snapshot restore failures stop training; they are not silently skipped.
- Dataset split overlap and validation/training seed overlap fail during startup.

## Testing And Acceptance

Unit tests cover:

- immutable base noise after repeated assignments;
- effect-aligned horizon and token-zero frame mapping;
- raw-ratio gradients while applied clipping is saturated;
- FIFO eviction and weighted sampling;
- deterministic dynamic-seed replay after checkpoint restore;
- dataset sample limiting and split identity;
- checkpoint round trip;
- optimizer ownership and frozen LDF/VAE gradients.

Integration smoke tests cover:

- two training samples, one collection cycle, and two optimizer steps;
- replay tensors stored on CPU;
- validation alpha-zero equivalence;
- resume continuing from the exact next collector and sampler RNG state.

The 300-sample pilot is accepted when held-out validation motions improve mean
ADE/FDE without non-finite failures, alpha zero is equivalent to Gaussian, and
sample-level harm rate is reported. Motion-level generalization, not training
loss alone, determines whether the sample cap is removed.
