import os
import random
from typing import Dict, List

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.motion_process import extract_root_trajectory_263
from utils.traj_batch import path_heading_features_from_root_xyz


class LengthMismatchError(Exception):
    pass


class HumanML3DDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.stream_mode = cfg.data.get("stream_mode", False)
        # Trajectory observation sparsity on token timeline (train uses mask_ratio).
        # - float in (0,1]: keep exactly ratio of tokens
        # - (min,max): keep a random ratio uniformly in [min,max] per sample (backwards compatible)
        # val/test default to full trajectory (1.0) for metrics/video alignment with deployment;
        # set data.val_mask_ratio to override (e.g. match train sparsity for ablations).
        if self.split in ("val", "test"):
            self.mask_ratio = cfg.data.get("val_mask_ratio", 1.0)
        else:
            self.mask_ratio = cfg.data.get("mask_ratio", (0.2, 0.3))
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
                                if self.feature_path is not None:
                                    feature_path = os.path.join(
                                        data_path, self.feature_path, name + ".npy"
                                    )
                                    feature = self.load_feature(feature_path)
                                    data["feature"] = feature
                                    data["feature_length"] = feature.shape[0]
                                if self.token_path is not None:
                                    token_path = os.path.join(
                                        data_path,
                                        self.token_path,
                                        name + ".npy",
                                    )
                                    token = self.load_token(token_path)
                                    data["token"] = token
                                    data["token_length"] = token.shape[0]
                                if self.text_path is not None:
                                    text_path = os.path.join(
                                        data_path, self.text_path, name + ".txt"
                                    )
                                    text_data = self.load_text(text_path)
                                    data["text_data"] = text_data
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
                tmp = random.choice(dataset)
                rank_zero_info(f"Random data {tmp['name']}: {tmp['feature'].shape}")
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
        return self._process(data)

    def _process(self, data):
        output = {}
        output["dataset"] = data["dataset"]
        output["name"] = data["name"]
        # --- feature ---
        crop_start = 0
        feature_length = None
        if "feature" in data:
            feature, feature_length, crop_start = self.process_feature(data["feature"])
            output["feature"] = feature
            output["feature_length"] = feature_length
            # --- traj (derived from feature) ---
            traj = extract_root_trajectory_263(feature)
            output["traj"] = traj
            output["traj_length"] = len(traj)

        # --- token ---
        if "token" in data:
            token, token_length = self.process_token(
                data["token"],
                crop_start=crop_start if feature_length is not None else None,
                feature_length=feature_length,
            )
            output["token"] = token
            output["token_length"] = token_length
            # --- mask (token-level sparsity, expanded to frame-level) ---
            token_mask = self.sample_token_mask(token_length)
            output["token_mask"] = token_mask

            if "traj" in output:
                traj_mask = np.repeat(token_mask, 4).astype(np.float32)
                traj_length = output["traj_length"]
                if traj_mask.shape[0] < traj_length:
                    pad = np.zeros((traj_length - traj_mask.shape[0],), dtype=np.float32)
                    traj_mask = np.concatenate([traj_mask, pad], axis=0)
                else:
                    traj_mask = traj_mask[:traj_length]
                output["traj_mask"] = traj_mask

        # --- traj_features: [x, z, cos(psi), sin(psi)], psi = xz path heading ---
        if "feature" in output and "token" in output:
            traj_features = path_heading_features_from_root_xyz(output["traj"])
            output["traj_features"] = traj_features

        # --- text ---
        if "text_data" in data:
            text_dict = random.choice(data["text_data"])
            text, text_tokens, f_tag, to_tag = self.process_text_dict(text_dict)
            if self.stream_mode:
                output["text"] = [text]
                output["text_tokens"] = text_tokens
                output["feature_text_end"] = [output["feature_length"]]
                output["token_text_end"] = [output["token_length"]]
            else:
                output["text"] = text
                output["text_tokens"] = text_tokens
        return output

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
        """Process token. When crop_start and feature_length are provided (from feature crop),
        crop token to align with feature (VAE temporal factor 4). Otherwise use original random crop."""
        token_length = len(token)
        if crop_start is not None and feature_length is not None:
            # Align token with feature: token_start = crop_start//4, token_len = feature_length//4
            token_start = crop_start // 4
            token_len = feature_length // 4
            end = min(token_start + token_len, token_length)
            if token_start >= token_length or token_len <= 0:
                token = token[:1]  # fallback: at least 1 token
            else:
                token = token[token_start:end]
        elif token_length > self.random_length:
            new_token_length = token_length - random.randint(0, self.random_length)
            start = random.randint(0, token_length - new_token_length)
            token = token[start : start + new_token_length]
        token_length = len(token)
        return token, token_length

    def process_text_dict(self, text_dict: Dict):
        text = text_dict["caption"]
        text_tokens = text_dict["tokens"]
        f_tag = text_dict["f_tag"]
        to_tag = text_dict["to_tag"]
        return text, text_tokens, f_tag, to_tag

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
        if key in ["feature", "token", "traj", "traj_features"]:
            # Pad sequences
            items = [
                torch.from_numpy(b[key]) if isinstance(b[key], np.ndarray) else b[key]
                for b in batch
            ]
            output[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0
            )
        elif key in ["traj_mask", "token_mask"]:
            # Pad traj_mask to (B, T_max), padding 填 0
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
    dataset = HumanML3DDataset(cfg, split="val")
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.data.train_bs, shuffle=True, collate_fn=collate_fn
    )
    for idx, data in tqdm(enumerate(dataloader)):
        pass
