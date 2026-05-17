from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


def build_stream_step_model_input(
    current_text: str, traj_input: Optional[Dict] = None
) -> Dict:
    payload = {"text": [str(current_text)]}
    if traj_input:
        payload.update(traj_input)
    return payload


@dataclass(frozen=True)
class StreamTextSegment:
    text: str
    token_end: int
    feature_end: Optional[int] = None


class StreamTextRolloutController:
    def __init__(self, segments: Sequence[StreamTextSegment]):
        if len(segments) == 0:
            segments = [StreamTextSegment(text="", token_end=0, feature_end=0)]
        cleaned: List[StreamTextSegment] = []
        prev_token_end = -1
        for segment in segments:
            token_end = int(segment.token_end)
            if token_end < prev_token_end:
                raise ValueError("token_end must be non-decreasing")
            cleaned.append(
                StreamTextSegment(
                    text=str(segment.text),
                    token_end=token_end,
                    feature_end=None
                    if segment.feature_end is None
                    else int(segment.feature_end),
                )
            )
            prev_token_end = token_end
        self.segments = cleaned

    @classmethod
    def from_sample_batch(cls, sample_batch: Dict):
        text_value = sample_batch.get("text", [""])
        if isinstance(text_value, list) and len(text_value) == 1:
            text_value = text_value[0]
        token_end_value = sample_batch.get("token_text_end", [[0]])
        if isinstance(token_end_value, list) and len(token_end_value) == 1:
            token_end_value = token_end_value[0]
        feature_end_value = sample_batch.get("feature_text_end", [[0]])
        if isinstance(feature_end_value, list) and len(feature_end_value) == 1:
            feature_end_value = feature_end_value[0]

        if not isinstance(text_value, list):
            text_value = [str(text_value)]
        token_end_value = list(token_end_value)
        feature_end_value = list(feature_end_value)
        if len(token_end_value) != len(text_value):
            raise ValueError(
                f"text/token_text_end length mismatch: {len(text_value)} vs {len(token_end_value)}"
            )
        if feature_end_value and len(feature_end_value) != len(text_value):
            raise ValueError(
                f"text/feature_text_end length mismatch: {len(text_value)} vs {len(feature_end_value)}"
            )

        segments = []
        for idx, text in enumerate(text_value):
            feature_end = feature_end_value[idx] if idx < len(feature_end_value) else None
            segments.append(
                StreamTextSegment(
                    text=str(text),
                    token_end=int(token_end_value[idx]),
                    feature_end=None if feature_end is None else int(feature_end),
                )
            )
        return cls(segments)

    def get_text_for_commit_index(self, commit_index: int) -> str:
        commit_index = int(commit_index)
        for segment in self.segments:
            if commit_index < segment.token_end:
                return segment.text
        return self.segments[-1].text


def build_stream_suffix_conditioning(sample_batch: Dict, commit_index: int) -> Dict:
    payload: Dict = {}
    token_mask = sample_batch.get("token_mask", None)
    if token_mask is not None:
        payload["token_mask"] = _slice_batch_suffix(token_mask, commit_index)

    traj = sample_batch.get("traj", None)
    if traj is not None:
        payload["traj"] = _slice_batch_suffix(
            _to_token_level_traj(sample_batch), commit_index
        )
        return payload

    traj_features = sample_batch.get("traj_features", None)
    if traj_features is not None:
        payload["traj_features"] = _slice_batch_suffix(
            _to_token_level_traj_features(sample_batch), commit_index
        )
    return payload


def _slice_batch_suffix(value, commit_index: int):
    if torch.is_tensor(value):
        if value.ndim == 1:
            return value[commit_index:]
        if value.ndim >= 2:
            return value[:, commit_index:]
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return value[commit_index:]
        if value.ndim >= 2:
            return value[:, commit_index:]
        return value
    return value


def _to_token_level_traj(sample_batch: Dict):
    traj = sample_batch["traj"]
    token_mask = sample_batch.get("token_mask", None)
    if token_mask is not None and torch.is_tensor(traj) and torch.is_tensor(token_mask):
        if traj.ndim >= 3 and token_mask.ndim >= 2 and traj.shape[1] == token_mask.shape[1]:
            return traj
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj).float()
    if traj.ndim == 2:
        traj = traj.unsqueeze(0)
    token_length = sample_batch.get("token_length", None)
    if token_length is None:
        raise ValueError("sample_batch must include token_length to build token-level traj")
    batch_size = traj.shape[0]
    token_xyz = []
    for batch_idx in range(batch_size):
        num_tokens = int(token_length[batch_idx].item()) if torch.is_tensor(token_length) else int(token_length[batch_idx])
        frame_indices = [0] + [min(4 * token_idx, traj.shape[1] - 1) for token_idx in range(1, num_tokens)]
        token_xyz.append(traj[batch_idx, frame_indices, :])
    return torch.stack(token_xyz, dim=0)


def _to_token_level_traj_features(sample_batch: Dict):
    traj_features = sample_batch["traj_features"]
    token_mask = sample_batch.get("token_mask", None)
    if token_mask is not None and torch.is_tensor(traj_features) and torch.is_tensor(token_mask):
        if traj_features.ndim >= 3 and token_mask.ndim >= 2 and traj_features.shape[1] == token_mask.shape[1]:
            return traj_features
    if isinstance(traj_features, np.ndarray):
        traj_features = torch.from_numpy(traj_features).float()
    if traj_features.ndim == 2:
        traj_features = traj_features.unsqueeze(0)
    token_length = sample_batch.get("token_length", None)
    if token_length is None:
        raise ValueError("sample_batch must include token_length to build token-level traj_features")
    batch_size = traj_features.shape[0]
    token_feats = []
    for batch_idx in range(batch_size):
        num_tokens = int(token_length[batch_idx].item()) if torch.is_tensor(token_length) else int(token_length[batch_idx])
        frame_indices = [0] + [min(4 * token_idx, traj_features.shape[1] - 1) for token_idx in range(1, num_tokens)]
        token_feats.append(traj_features[batch_idx, frame_indices, :])
    return torch.stack(token_feats, dim=0)
