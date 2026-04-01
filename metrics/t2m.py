from typing import List

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

from utils.initialize import instantiate

from .tools.utils import (
    calculate_activation_statistics_np,
    calculate_diversity_np,
    calculate_frechet_distance_np,
    calculate_top_k,
    euclidean_distance_matrix,
)


class T2MMetrics(Metric):
    def __init__(
        self,
        cfg,
        top_k=3,
        R_size=32,
        diversity_times=300,
        dist_sync_on_step=False,
    ):
        # Disable automatic distributed sync since we manually handle it in compute()
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=False)

        self.cfg = cfg
        self.evaluate_text = self.cfg.evaluate_text

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        # Metric names list
        self.metrics = []

        if self.evaluate_text:
            self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
            for k in range(1, top_k + 1):
                self.Matching_metrics.append(f"R_precision_top_{str(k)}")
            for k in range(1, top_k + 1):
                self.Matching_metrics.append(f"gt_R_precision_top_{str(k)}")
            self.metrics.extend(self.Matching_metrics)

        self.metrics.extend(["FID", "Diversity", "gt_Diversity"])

        # Chached batches
        self.add_state("text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("recmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)

        # T2M Evaluator
        self._get_t2m_evaluator(self.cfg)

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        t2m_checkpoint = {}
        if self.evaluate_text:
            self.w_vectorizer = instantiate(
                cfg.wordvectorizer.target,
                cfg=None,
                **cfg.wordvectorizer.params,
            )
            self.t2m_textencoder = instantiate(
                cfg.textencoder.target, cfg=None, **cfg.textencoder.params
            )
            t2m_checkpoint["text_encoder"] = torch.load(
                cfg.textencoder.ckpt,
                map_location="cpu",
            )
            self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
            self.t2m_textencoder.eval()
            for p in self.t2m_textencoder.parameters():
                p.requires_grad = False

        self.metric_mean_np = np.load(cfg.metric_mean_path)
        self.metric_std_np = np.load(cfg.metric_std_path)

        # Convert to torch tensors immediately to avoid repeated conversion
        self.register_buffer(
            "metric_mean", torch.from_numpy(self.metric_mean_np).float()
        )
        self.register_buffer("metric_std", torch.from_numpy(self.metric_std_np).float())

        self.t2m_moveencoder = instantiate(
            cfg.moveencoder.target, cfg=None, **cfg.moveencoder.params
        )
        self.t2m_motionencoder = instantiate(
            cfg.motionencoder.target, cfg=None, **cfg.motionencoder.params
        )

        t2m_checkpoint["movement_encoder"] = torch.load(
            cfg.moveencoder.ckpt,
            map_location="cpu",
        )
        t2m_checkpoint["motion_encoder"] = torch.load(
            cfg.motionencoder.ckpt,
            map_location="cpu",
        )

        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()

        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute(self, sanity_flag):
        # Init empty metrics dict (will be filled with computed values)
        metrics = {}

        # Use current device for metrics (CUDA if available, to match model device)
        metrics_device = self.device

        # Jump in sanity check stage and no embeddings
        if sanity_flag or not self.recmotion_embeddings:
            # Return dummy values for sanity check
            if self.evaluate_text:
                for metric in self.Matching_metrics:
                    metrics[metric] = torch.tensor(0.0, device=metrics_device)
            metrics["FID"] = torch.tensor(0.0, device=metrics_device)
            metrics["Diversity"] = torch.tensor(0.0, device=metrics_device)
            metrics["gt_Diversity"] = torch.tensor(0.0, device=metrics_device)
            return metrics

        # Gather embeddings from all GPUs
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()

            # Gather embeddings from all ranks (already on CPU from update())
            gathered_recmotion = [None] * world_size
            gathered_gtmotion = [None] * world_size

            torch.distributed.all_gather_object(
                gathered_recmotion, self.recmotion_embeddings
            )
            torch.distributed.all_gather_object(
                gathered_gtmotion, self.gtmotion_embeddings
            )

            # Flatten the gathered lists (no need to move to CPU, already there)
            self.recmotion_embeddings = [
                emb for rank_embs in gathered_recmotion for emb in rank_embs
            ]
            self.gtmotion_embeddings = [
                emb for rank_embs in gathered_gtmotion for emb in rank_embs
            ]

            if self.evaluate_text:
                gathered_text = [None] * world_size
                torch.distributed.all_gather_object(gathered_text, self.text_embeddings)
                self.text_embeddings = [
                    emb for rank_embs in gathered_text for emb in rank_embs
                ]

        # Now use the global embeddings (already on CPU)
        count_seq = len(self.recmotion_embeddings)

        # Cat cached batches and shuffle
        shuffle_idx = torch.randperm(count_seq)
        all_genmotions = torch.cat(self.recmotion_embeddings, axis=0)[shuffle_idx, :]
        all_gtmotions = torch.cat(self.gtmotion_embeddings, axis=0)[shuffle_idx, :]

        # Compute text related metrics
        if self.evaluate_text:
            all_texts = torch.cat(self.text_embeddings, axis=0)[shuffle_idx, :]

            if count_seq > self.R_size:
                # Compute r-precision
                # Reset and recompute matching scores from global data
                matching_score_sum = 0.0
                top_k_mat = torch.zeros((self.top_k,))
                for i in range(count_seq // self.R_size):
                    # [bs=32, 1*256]
                    group_texts = all_texts[i * self.R_size : (i + 1) * self.R_size]
                    # [bs=32, 1*256]
                    group_motions = all_genmotions[
                        i * self.R_size : (i + 1) * self.R_size
                    ]
                    # dist_mat = pairwise_euclidean_distance(group_texts, group_motions)
                    # [bs=32, 32]
                    dist_mat = euclidean_distance_matrix(
                        group_texts, group_motions
                    ).nan_to_num()
                    # print(dist_mat[:5])
                    # PyTorch trace does not support BFloat16, convert to float32
                    if dist_mat.dtype == torch.bfloat16:
                        dist_mat = dist_mat.float()
                    matching_score_sum += dist_mat.trace()
                    argsmax = torch.argsort(dist_mat, dim=1)
                    top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)

                R_count = count_seq // self.R_size * self.R_size
                metrics["Matching_score"] = (
                    (matching_score_sum / R_count).detach().clone().to(metrics_device)
                )
                for k in range(self.top_k):
                    metrics[f"R_precision_top_{str(k + 1)}"] = (
                        (top_k_mat[k] / R_count).detach().clone().to(metrics_device)
                    )

                # Compute r-precision with gt
                gt_matching_score_sum = 0.0
                top_k_mat = torch.zeros((self.top_k,))
                for i in range(count_seq // self.R_size):
                    # [bs=32, 1*256]
                    group_texts = all_texts[i * self.R_size : (i + 1) * self.R_size]
                    # [bs=32, 1*256]
                    group_motions = all_gtmotions[
                        i * self.R_size : (i + 1) * self.R_size
                    ]
                    # [bs=32, 32]
                    dist_mat = euclidean_distance_matrix(
                        group_texts, group_motions
                    ).nan_to_num()
                    # match score
                    # PyTorch trace does not support BFloat16, convert to float32
                    if dist_mat.dtype == torch.bfloat16:
                        dist_mat = dist_mat.float()
                    gt_matching_score_sum += dist_mat.trace()
                    argsmax = torch.argsort(dist_mat, dim=1)
                    top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
                metrics["gt_Matching_score"] = (
                    (gt_matching_score_sum / R_count)
                    .detach()
                    .clone()
                    .to(metrics_device)
                )
                for k in range(self.top_k):
                    metrics[f"gt_R_precision_top_{str(k + 1)}"] = (
                        (top_k_mat[k] / R_count).detach().clone().to(metrics_device)
                    )
            else:
                for metric in self.Matching_metrics:
                    metrics[metric] = torch.tensor(0.0, device=metrics_device)

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.float().numpy()
        all_gtmotions = all_gtmotions.float().numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        fid_value = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)
        metrics["FID"] = torch.tensor(
            fid_value, dtype=torch.float32, device=metrics_device
        )

        # Compute diversity
        if count_seq > self.diversity_times:
            diversity_value = calculate_diversity_np(
                all_genmotions, self.diversity_times
            )
            gt_diversity_value = calculate_diversity_np(
                all_gtmotions, self.diversity_times
            )
            metrics["Diversity"] = torch.tensor(
                diversity_value, dtype=torch.float32, device=metrics_device
            )
            metrics["gt_Diversity"] = torch.tensor(
                gt_diversity_value, dtype=torch.float32, device=metrics_device
            )

        # Ensure all metrics are on the correct device
        for key in metrics:
            if (
                isinstance(metrics[key], torch.Tensor)
                and metrics[key].device != metrics_device
            ):
                metrics[key] = metrics[key].to(metrics_device)

        # Reset
        self.reset()
        return metrics

    @torch.no_grad()
    def update(
        self,
        feats_ref: Tensor,
        feats_rst: Tensor,
        lengths_ref: List[int],
        lengths_rst: List[int],
        text_tokens: List[List[str]] = None,
    ):
        # T2m motion encoder
        align_idx = np.argsort(lengths_ref)[::-1].copy()
        feats_ref = feats_ref[align_idx]
        lengths_ref = np.array(lengths_ref)[align_idx]
        gtmotion_embeddings = self.get_motion_embeddings(feats_ref, lengths_ref)
        cache = [0] * len(lengths_ref)
        for i in range(len(lengths_ref)):
            # Move to CPU to avoid accumulating GPU memory and device mismatch in DDP
            cache[align_idx[i]] = gtmotion_embeddings[i : i + 1].cpu()
        self.gtmotion_embeddings.extend(cache)

        align_idx = np.argsort(lengths_rst)[::-1].copy()
        feats_rst = feats_rst[align_idx]
        lengths_rst = np.array(lengths_rst)[align_idx]
        recmotion_embeddings = self.get_motion_embeddings(feats_rst, lengths_rst)
        cache = [0] * len(lengths_rst)
        for i in range(len(lengths_rst)):
            # Move to CPU to avoid accumulating GPU memory and device mismatch in DDP
            cache[align_idx[i]] = recmotion_embeddings[i : i + 1].cpu()
        self.recmotion_embeddings.extend(cache)

        # T2m text encoder
        if self.evaluate_text:
            # Process each text entry
            for tokens in text_tokens:
                max_text_len = self.cfg.wordvectorizer.max_text_len
                if len(tokens) < max_text_len:
                    # pad with "unk"
                    tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                    sent_len = len(tokens)
                    tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
                else:
                    # crop
                    tokens = tokens[:max_text_len]
                    tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                    sent_len = len(tokens)

                pos_one_hots = []
                word_embeddings = []
                for token in tokens:
                    word_emb, pos_oh = self.w_vectorizer[token]
                    pos_one_hots.append(pos_oh[None, :])
                    word_embeddings.append(word_emb[None, :])
                pos_one_hots = np.concatenate(pos_one_hots, axis=0)
                word_embeddings = np.concatenate(word_embeddings, axis=0)

                # Convert to tensor and move to correct device
                pos_one_hots = torch.from_numpy(pos_one_hots.astype(np.float32)).to(
                    self.device
                )
                word_embeddings = torch.from_numpy(
                    word_embeddings.astype(np.float32)
                ).to(self.device)
                text_lengths = torch.tensor(sent_len).to(self.device)

                # Encode
                # Add batch dim [1, ...]
                self.t2m_textencoder = self.t2m_textencoder.to(self.device)
                text_emb = self.t2m_textencoder(
                    word_embeddings[None, ...],
                    pos_one_hots[None, ...],
                    text_lengths[None, ...],
                )
                text_embeddings = torch.flatten(text_emb, start_dim=1).detach()
                # Move to CPU
                self.text_embeddings.append(text_embeddings.cpu())

    def get_motion_embeddings(self, feats: Tensor, lengths: List[int]):
        # Ensure models are on the same device as input
        device = feats.device
        feats = (feats - self.metric_mean) / self.metric_std

        self.t2m_moveencoder = self.t2m_moveencoder.to(device)
        self.t2m_motionencoder = self.t2m_motionencoder.to(device)

        m_lens = torch.tensor(lengths)
        m_lens = torch.div(m_lens, 4, rounding_mode="floor")
        mov = self.t2m_moveencoder(feats[..., :-4]).detach()
        emb = self.t2m_motionencoder(mov, m_lens)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        return torch.flatten(emb, start_dim=1).detach()
