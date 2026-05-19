import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.motion_process import extract_root_trajectory_263
from utils.traj_batch import root_to_traj_feats, smooth_root_xz


class LengthMismatchError(Exception):
    pass


class BabelDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        if self.split in ("val", "test"):
            self.mask_ratio = cfg.data.get("val_mask_ratio", 1.0)
        else:
            self.mask_ratio = cfg.data.get("mask_ratio", 1.0)
        if self.split == "train":
            self.file_list = cfg.data.train_meta_paths
            self.min_length = cfg.data.min_length
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.get("window_length", cfg.data.max_length)
            self.random_length = cfg.data.get("random_length", 0)
        elif self.split == "val":
            self.file_list = cfg.data.val_meta_paths
            self.min_length = cfg.data.min_length
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.max_length
            self.random_length = 0
        elif self.split == "test":
            self.file_list = cfg.data.test_meta_paths
            self.min_length = 0
            self.max_length = cfg.data.max_length
            self.window_length = cfg.data.max_length
            self.random_length = 0
        self.feature_path = cfg.data.get("feature_path", None)
        self.token_path = cfg.data.get("token_path", None)
        self.text_path = cfg.data.get("text_path", None)
        self.feature_fps = float(cfg.data.get("feature_fps", 20))
        self.smooth_traj_sigma = float(cfg.data.get("smooth_traj_sigma", 0.0))
        self.dataset = self._load_file_list()

        rank_zero_info(f"Loaded {len(self.dataset)} samples from {split} dataset.")

    def _load_file_list(self) -> List[str]:
        """Load a list of npy file paths from a text file."""
        dataset = []
        ignored_cnt = 0
        for path in self.file_list:
            if os.path.exists(path):
                data_path = os.path.dirname(path)
                dataset_name = os.path.basename(data_path)
                rank_zero_info(f"Loading {path} ...")
                with open(path, "r") as f:
                    for name in f:
                        name = name.strip()
                        if name:
                            data = {}
                            try:
                                data["name"] = name
                                data["dataset"] = dataset_name
                                ##############################
                                # feature
                                ##############################
                                if self.feature_path is not None:
                                    feature_path = os.path.join(
                                        data_path, self.feature_path, name + ".npy"
                                    )
                                    feature = self.load_feature(feature_path)
                                    data["feature"] = feature
                                    data["feature_length"] = feature.shape[0]
                                ##############################
                                # token
                                ##############################
                                if self.token_path is not None:
                                    token_path = os.path.join(
                                        data_path,
                                        self.token_path,
                                        name + ".npy",
                                    )
                                    token = self.load_token(token_path)
                                    data["token"] = token
                                    data["token_length"] = token.shape[0]
                                ##############################
                                # text
                                ##############################
                                if self.text_path is not None:
                                    text_path = os.path.join(
                                        data_path, self.text_path, name + ".txt"
                                    )
                                    text_data = self.load_text(text_path)
                                    data["text_data"] = text_data
                                ##############################
                                # end
                                ##############################
                                dataset.append(data)
                            except LengthMismatchError:
                                ignored_cnt += 1
                                pass
                            except Exception as e:
                                rank_zero_info(f"Error loading data for {name}: {e}")
                                pass
                            if self.cfg.debug and len(dataset) >= 100:
                                rank_zero_info(f"debug mode, break at {len(dataset)}")
                                break
        if ignored_cnt > 0:
            rank_zero_info(f"Ignored {ignored_cnt} samples due to length mismatch.")
        if len(dataset) == 0:
            rank_zero_info(
                f"No data found in {self.file_list}. Please check the file paths and ensure they are correct."
            )
        else:
            for i in range(3):
                tmp = dataset[i % len(dataset)]  # avoid burning global RNG
                shape_str = tmp["feature"].shape if "feature" in tmp else "(no feature)"
                rank_zero_info(f"Random data {tmp['name']}: {shape_str}")
        return dataset

    def load_feature(self, feature_path: str) -> np.ndarray:
        feature = np.load(feature_path)
        if np.isnan(feature).any():
            raise ValueError("NaN values found in feature, skip it.")
        if feature.shape[0] < self.min_length or feature.shape[0] > self.max_length:
            raise LengthMismatchError("Feature length out of bounds, skip it.")
        return feature

    def load_token(self, token_path: str) -> np.ndarray:
        token = np.load(token_path)
        if np.isnan(token).any():
            raise ValueError("NaN values found in token, skip it.")
        return token

    def load_text(self, text_path: str) -> List[Dict]:
        text_data = []
        with open(text_path, "r") as text_f:
            lines = text_f.readlines()
            for line in lines:
                text_dict = {}
                line_split = line.strip().split("#")
                caption = line_split[0].strip()
                t_tokens = line_split[1].split(" ")
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                text_dict["caption"] = caption
                text_dict["tokens"] = t_tokens
                text_dict["f_tag"] = f_tag
                text_dict["to_tag"] = to_tag

                text_data.append(text_dict)
        return text_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return self._build_output(data, apply_crop=True)

    def _build_output(self, data, apply_crop: bool):
        output = {}
        output["dataset"] = data["dataset"]
        output["name"] = data["name"]
        if "segment_names" in data:
            output["segment_names"] = list(data["segment_names"])
        ##############################
        # feature
        ##############################
        crop_start = 0
        feature_length = None
        if "feature" in data:
            if apply_crop:
                feature, feature_length, crop_start = self.process_feature(data["feature"])
            else:
                feature = data["feature"]
                feature_length = feature.shape[0]
            output["feature"] = feature
            output["feature_length"] = feature_length
            ##############################
            # traj (root xyz derived from feature)
            ##############################
            traj = extract_root_trajectory_263(feature)
            output["traj"] = traj
            output["traj_cond"] = traj
            output["traj_loss_gt"] = traj
            output["traj_length"] = len(traj)

        ##############################
        # token
        ##############################
        if "token" in data:
            if apply_crop:
                token, token_length, token_start = self.process_token(
                    data["token"],
                    crop_start=crop_start if feature_length is not None else None,
                    feature_length=feature_length,
                )
            else:
                token = data["token"]
                token_length = len(token)
                token_start = 0
            output["token"] = token
            output["token_length"] = token_length
            ##############################
            # mask (token-level sparsity expanded to frame-level VAE)
            ##############################
            token_mask = self.sample_token_mask(token_length)
            output["token_mask"] = token_mask

            if "traj" in output:
                # Causal VAE: token 0 → frame 0; token k≥1 → frames [4k-3, 4k]
                traj_length = output["traj_length"]
                traj_mask = np.zeros(traj_length, dtype=np.float32)
                if len(token_mask) > 0:
                    traj_mask[0] = token_mask[0]
                for k in range(1, len(token_mask)):
                    sf = 4 * k - 3
                    ef = min(4 * k + 1, traj_length)
                    if sf < traj_length:
                        traj_mask[sf:ef] = token_mask[k]
                output["traj_mask"] = traj_mask
                output["traj_cond_mask"] = traj_mask.copy()
                output["traj_loss_mask"] = traj_mask.copy()
        else:
            token_start = 0

        ##############################
        # traj_features
        ##############################
        # [x, z, cos(ψ), sin(ψ)] for ControlNet conditioning.
        # output["traj"] (raw xyz) feeds L_control_xz GT supervision.
        if "feature" in output and "token" in output:
            traj_xyz = output["traj"]
            if self.smooth_traj_sigma > 0.0:
                traj_xz_smooth = smooth_root_xz(
                    traj_xyz[:, [0, 2]], sigma=self.smooth_traj_sigma
                )
                traj_xyz_for_cond = traj_xyz.copy()
                traj_xyz_for_cond[:, 0] = traj_xz_smooth[:, 0]
                traj_xyz_for_cond[:, 2] = traj_xz_smooth[:, 1]
            else:
                traj_xyz_for_cond = traj_xyz
            output["traj_features"] = root_to_traj_feats(traj_xyz_for_cond)

        ##############################
        # text
        ##############################
        if "text_data" in data:
            (
                texts,
                text_token_lists,
                feature_text_end,
                token_text_end,
            ) = self.process_text_data(
                data["text_data"],
                crop_start=crop_start,
                feature_length=feature_length if feature_length is not None else data.get("feature_length", 0),
                token_length=output.get("token_length", 0),
                token_start=token_start,
            )
            output["text"] = texts
            output["text_tokens"] = next(
                (tokens for tokens in reversed(text_token_lists) if len(tokens) > 0),
                [],
            )
            output["feature_text_end"] = feature_text_end
            output["token_text_end"] = token_text_end
        return output

    def build_present_segment_eval_samples(self) -> List[Dict]:
        grouped: Dict[Tuple[str, str], List[Tuple[int, Dict]]] = defaultdict(list)
        for data in self.dataset:
            base_name, order = self.parse_present_segment_name(data["name"])
            grouped[(data["dataset"], base_name)].append((order, data))

        merged_samples: List[Dict] = []
        for (_, base_name), entries in grouped.items():
            entries.sort(key=lambda item: (item[0], item[1]["name"]))
            merged_raw = self._merge_present_segment_group(
                base_name, [item[1] for item in entries]
            )
            merged_samples.append(self._build_output(merged_raw, apply_crop=False))
        return merged_samples

    def _merge_present_segment_group(self, base_name: str, group: List[Dict]) -> Dict:
        merged = {
            "dataset": group[0]["dataset"],
            "name": base_name,
            "segment_names": [item["name"] for item in group],
        }

        if "feature" in group[0]:
            merged["feature"] = np.concatenate(
                [item["feature"] for item in group], axis=0
            ).astype(np.float32, copy=False)
        if "token" in group[0]:
            merged["token"] = np.concatenate(
                [item["token"] for item in group], axis=0
            ).astype(np.float32, copy=False)
        if "text_data" in group[0]:
            merged["text_data"] = self._merge_segment_text_data(group)
        return merged

    def _merge_segment_text_data(self, group: List[Dict]) -> List[Dict]:
        merged_text_data: List[Dict] = []
        feature_offset_frames = 0
        for item in group:
            feature = item.get("feature", None)
            if feature is None:
                continue
            segment_frames = int(feature.shape[0])
            segment_seconds = float(segment_frames) / self.feature_fps
            offset_seconds = float(feature_offset_frames) / self.feature_fps

            for text_dict in item.get("text_data", []):
                text = text_dict["caption"]
                text_tokens = list(text_dict["tokens"])
                f_tag = float(text_dict["f_tag"])
                to_tag = float(text_dict["to_tag"])
                if f_tag == 0.0 and to_tag == 0.0:
                    abs_f_tag = offset_seconds
                    abs_to_tag = offset_seconds + segment_seconds
                else:
                    abs_f_tag = offset_seconds + f_tag
                    abs_to_tag = offset_seconds + to_tag
                merged_text_data.append(
                    {
                        "caption": text,
                        "tokens": text_tokens,
                        "f_tag": abs_f_tag,
                        "to_tag": abs_to_tag,
                    }
                )
            feature_offset_frames += segment_frames
        return merged_text_data

    @staticmethod
    def parse_present_segment_name(name: str) -> Tuple[str, int]:
        if "_" not in name:
            return name, 0
        head, tail = name.rsplit("_", 1)
        if tail.isdigit():
            return head, int(tail)
        return name, 0

    def process_feature(self, feature):
        feature_len = feature.shape[0]
        crop_start = 0
        # if the motion is longer than window_length, randomly crop a window_length clip
        if feature_len > self.window_length:
            crop_start = random.randint(0, feature_len - self.window_length)
            feature = feature[crop_start : crop_start + self.window_length]
            feature_len = self.window_length
        return feature, feature_len, crop_start

    def process_token(self, token, crop_start=None, feature_length=None):
        token_length = len(token)
        token_start = 0
        if crop_start is not None and feature_length is not None:
            token_start = (crop_start + 3) // 4
            last_frame = crop_start + feature_length - 1
            token_end = (last_frame + 3) // 4  # inclusive
            token_len = token_end - token_start + 1
            end = min(token_start + token_len, token_length)
            if token_start >= token_length or token_len <= 0:
                token = token[:1]
                token_start = 0
            else:
                token = token[token_start:end]
        elif token_length > self.random_length:
            new_token_length = token_length - random.randint(0, self.random_length)
            start = random.randint(0, token_length - new_token_length)
            token = token[start : start + new_token_length]
            token_start = start
        token_length = len(token)
        return token, token_length, token_start

    def process_text_data(
        self,
        text_data: List[Dict],
        crop_start: int,
        feature_length: int,
        token_length: int,
        token_start: int,
    ):
        if feature_length <= 0:
            return [""], [[]], [0], [0]

        crop_end = crop_start + feature_length
        segments = []
        for text_dict in text_data:
            text, text_tokens, f_tag, to_tag = self.process_text_dict(text_dict)
            abs_start = max(0, int(f_tag * self.feature_fps + 0.5))
            abs_end = int(to_tag * self.feature_fps + 0.5)
            if abs_end <= abs_start:
                if abs_start == 0 and abs_end == 0:
                    abs_end = crop_end
                else:
                    continue
            if abs_end <= crop_start or abs_start >= crop_end:
                continue
            rel_start = max(abs_start, crop_start) - crop_start
            rel_end = min(abs_end, crop_end) - crop_start
            if rel_end <= rel_start:
                continue
            segments.append((rel_start, rel_end, text, text_tokens))

        segments.sort(key=lambda item: (item[0], item[1]))

        texts = []
        text_token_lists = []
        feature_text_end = []
        cursor = 0
        for rel_start, rel_end, text, text_tokens in segments:
            if rel_start > cursor:
                texts.append("")
                text_token_lists.append([])
                feature_text_end.append(rel_start)
                cursor = rel_start
            if rel_end <= cursor:
                continue
            texts.append(text)
            text_token_lists.append(text_tokens)
            feature_text_end.append(rel_end)
            cursor = rel_end
        if cursor < feature_length or not texts:
            texts.append("")
            text_token_lists.append([])
            feature_text_end.append(feature_length)

        token_text_end = [
            self.frame_count_to_token_count(end_frame, token_length, token_start, crop_start)
            for end_frame in feature_text_end
        ]
        return texts, text_token_lists, feature_text_end, token_text_end

    def process_text_dict(self, text_dict: Dict):
        text = text_dict["caption"]
        text_tokens = text_dict["tokens"]
        f_tag = text_dict["f_tag"]
        to_tag = text_dict["to_tag"]
        return text, text_tokens, f_tag, to_tag

    @staticmethod
    def frame_count_to_token_count(
        feature_end: int, token_length: int, token_start: int, crop_start: int
    ) -> int:
        if token_length <= 0 or feature_end <= 0:
            return 0
        last_frame_abs = crop_start + feature_end - 1
        token_end_abs = (last_frame_abs + 3) // 4
        rel_count = token_end_abs - token_start + 1
        return max(0, min(token_length, rel_count))

    def sample_token_mask(self, token_length: int) -> np.ndarray:
        if token_length <= 0:
            return np.zeros((0,), dtype=np.float32)
        mask = np.zeros(token_length, dtype=np.float32)
        r = self.mask_ratio
        if isinstance(r, (list, tuple, ListConfig)) and len(r) == 2:
            r0, r1 = float(r[0]), float(r[1])
            keep_ratio = random.uniform(min(r0, r1), max(r0, r1))
        else:
            keep_ratio = float(r)
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        n_keep = max(1, int(round(token_length * keep_ratio)))
        indices = random.sample(range(token_length), n_keep)
        mask[indices] = 1.0
        return mask


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    output = {}
    keys = batch[0].keys()

    for key in keys:
        if key in ["feature", "token", "traj", "traj_cond", "traj_loss_gt",
                   "traj_features"]:
            # Pad sequences
            items = [
                torch.from_numpy(b[key]) if isinstance(b[key], np.ndarray) else b[key]
                for b in batch
            ]
            output[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0
            )
        elif key in ["traj_mask", "traj_cond_mask", "traj_loss_mask", "token_mask"]:
            items = [
                torch.from_numpy(b[key])
                if isinstance(b[key], np.ndarray)
                else torch.tensor(b[key], dtype=torch.float32)
                for b in batch
            ]
            output[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0
            )
        elif key in ["feature_length", "token_length", "traj_length"]:
            # Stack scalars
            output[key] = torch.tensor([b[key] for b in batch])
        else:
            # Default to list
            output[key] = [b[key] for b in batch]
    return output


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/default.yaml")
    dataset = BabelDataset(cfg, split="val")
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.data.train_bs, shuffle=True, collate_fn=collate_fn
    )
    for idx, data in tqdm(enumerate(dataloader)):
        pass
