from __future__ import annotations

import pytest
import torch

from utils.conditions.ldf import LDFCondition, LDFTrainingCondition


def test_ldf_condition_accepts_prepared_model_inputs():
    condition = LDFCondition(
        text_context=["text"],
        text_null_context=[""],
        traj_emb=torch.zeros(1, 5, 4),
        traj_seq_lens=torch.tensor([5]),
        traj_token_mask=torch.ones(1, 5),
        seq_len=5,
        attn_len=5,
    )

    condition.validate(batch_size=1)


def test_ldf_condition_separates_latent_and_attention_lengths():
    condition = LDFCondition(text_context=["text"], seq_len=5, attn_len=8)

    assert condition.latent_len() == 5
    assert condition.attention_len() == 8


def test_ldf_condition_requires_seq_len_for_latent_length():
    condition = LDFCondition(text_context=["text"], attn_len=8)

    with pytest.raises(ValueError, match="seq_len"):
        condition.latent_len()


def test_ldf_condition_rejects_mismatched_traj_batch_size():
    condition = LDFCondition(
        text_context=["text"],
        traj_emb=torch.zeros(2, 5, 4),
        traj_seq_lens=torch.tensor([5]),
        seq_len=5,
        attn_len=5,
    )

    with pytest.raises(ValueError, match="traj_emb batch size"):
        condition.validate(batch_size=1)


def test_training_condition_wraps_model_condition_without_polluting_contract():
    condition = LDFCondition(text_context=["text"], seq_len=3)
    training_condition = LDFTrainingCondition(
        condition=condition,
        text_dropped_flags=[False],
        traj_dropped=False,
    )

    assert training_condition.condition is condition
    assert not hasattr(condition, "text_dropped_flags")
    assert not hasattr(condition, "traj_dropped")
