"""LDF eval case/path contracts."""

from __future__ import annotations

from eval.common import EvalPathSpec


HUMANML3D_ONLY_DATASET = "humanml3d"

STREAM_GT = "stream_gt"
OFFLINE_GT = "offline_gt"
STREAM_NO_TRAJ = "stream_no_traj"
OFFLINE_NO_TRAJ = "offline_no_traj"

DEFAULT_LDF_EVAL_PATHS = (STREAM_GT, OFFLINE_GT, STREAM_NO_TRAJ)
OPTIONAL_LDF_EVAL_PATHS = (OFFLINE_NO_TRAJ,)

TEXT_BUCKET_MAX_SAMPLES = 5
TEXT_BUCKETS = (
    "walk",
    "run",
    "slow",
    "fast",
    "turn",
    "jump",
    "sit_stand",
    "dance",
    "other",
)

TEXT_BUCKET_KEYWORDS = {
    "walk": ("walk", "walking", "walks", "stroll", "strolling"),
    "run": ("run", "running", "runs", "jog", "jogging"),
    "slow": ("slow", "slowly"),
    "fast": ("fast", "quickly", "quick"),
    "turn": ("turn", "turning", "rotate", "spins", "spin"),
    "jump": ("jump", "jumping", "hop", "hops"),
    "sit_stand": ("sit", "sits", "sitting", "stand", "standing", "stands"),
    "dance": ("dance", "dancing"),
}


def build_ldf_path_specs(*, include_offline_no_traj: bool = False) -> list[EvalPathSpec]:
    specs = [
        EvalPathSpec(
            name=STREAM_GT,
            description="stream_generate_step + GT-7D",
            enabled_by_default=True,
            tags=("ldf", "stream", "gt_7d"),
        ),
        EvalPathSpec(
            name=OFFLINE_GT,
            description="offline generate + GT-7D",
            enabled_by_default=True,
            tags=("ldf", "offline", "gt_7d", "diagnostic"),
        ),
        EvalPathSpec(
            name=STREAM_NO_TRAJ,
            description="stream_generate_step + ControlNet(null)",
            enabled_by_default=True,
            tags=("ldf", "stream", "no_traj", "diagnostic"),
        ),
    ]
    if include_offline_no_traj:
        specs.append(
            EvalPathSpec(
                name=OFFLINE_NO_TRAJ,
                description="offline generate + ControlNet(null)",
                enabled_by_default=False,
                tags=("ldf", "offline", "no_traj", "diagnostic"),
            )
        )
    return specs


def classify_text_bucket(text: str) -> str:
    lowered = str(text).lower()
    for bucket in TEXT_BUCKETS:
        if bucket == "other":
            continue
        for keyword in TEXT_BUCKET_KEYWORDS.get(bucket, ()):
            if keyword in lowered:
                return bucket
    return "other"


__all__ = [
    "DEFAULT_LDF_EVAL_PATHS",
    "HUMANML3D_ONLY_DATASET",
    "OFFLINE_GT",
    "OFFLINE_NO_TRAJ",
    "OPTIONAL_LDF_EVAL_PATHS",
    "STREAM_GT",
    "STREAM_NO_TRAJ",
    "TEXT_BUCKET_KEYWORDS",
    "TEXT_BUCKET_MAX_SAMPLES",
    "TEXT_BUCKETS",
    "build_ldf_path_specs",
    "classify_text_bucket",
]
