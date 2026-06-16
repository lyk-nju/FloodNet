import torch

from omegaconf import OmegaConf
from utils.initialize import instantiate


def prepare_model_params(params):
    if OmegaConf.is_config(params):
        params = OmegaConf.to_container(params, resolve=True)
    else:
        params = dict(params)

    use_precomputed = bool(params.pop("use_precomputed_text_emb", False))
    precomputed_path = params.pop("precomputed_text_emb_path", None)
    if use_precomputed:
        if not precomputed_path:
            raise ValueError(
                "use_precomputed_text_emb=True requires precomputed_text_emb_path."
            )
        params["build_text_encoder"] = False
        return params, str(precomputed_path)
    return params, None


def expand_precomputed_caption_keys(embeddings: dict) -> dict:
    """Alias strip() keys so the table matches dataset captions after .strip()."""
    out = dict(embeddings)
    for key, value in embeddings.items():
        stripped_key = key.strip()
        if stripped_key not in out:
            out[stripped_key] = value
    return out


def load_precomputed_text_embeddings(path, *, expected_text_dim: int = 4096):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = expand_precomputed_caption_keys(payload["embeddings"])
    if "" not in embeddings:
        raise KeyError(
            'precomputed embeddings must include empty string key "" for CFG / dropout.'
        )
    text_dim = int(payload.get("text_dim", expected_text_dim))
    if text_dim != int(expected_text_dim):
        raise ValueError(
            f"precomputed text_dim {text_dim} != model text_dim {expected_text_dim}"
        )
    return embeddings


def install_precomputed_text_embeddings(
    model,
    path,
    *,
    expected_text_dim: int = 4096,
):
    model._precomputed_text_emb = load_precomputed_text_embeddings(
        path,
        expected_text_dim=expected_text_dim,
    )
    model.text_encoder = None
    return model


def instantiate_ldf_model(target, params):
    model_params, precomputed_path = prepare_model_params(params)
    model = instantiate(target=target, cfg=None, hfstyle=False, **model_params)
    if precomputed_path is not None:
        install_precomputed_text_embeddings(
            model,
            precomputed_path,
            expected_text_dim=getattr(model, "text_dim", 4096),
        )
    return model
