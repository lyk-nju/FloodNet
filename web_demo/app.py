"""Flask server bootstrap for the real-time 3D motion generation demo."""

import threading
import argparse
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask
from flask_cors import CORS
from web_demo.model_manager import get_model_manager
from web_demo.api.routes import register_routes
from web_demo.config import load_debug_preset_cfg, load_traj_mask_cfg
from web_demo.services.session_service import SessionService
from utils.motion_process import extract_root_trajectory_263
from utils.inference.geometry import resample_polyline

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
CORS(app)

# Global model manager (lazy loaded)
model_manager = None
model_config_path = None  # Will be set once at startup
traj_mask_cfg = None
debug_preset_cfg = None
model_init_lock = threading.Lock()
session_service = SessionService(consumption_timeout=5.0)


def init_model():
    """Initialize model manager"""
    global model_manager, traj_mask_cfg
    if model_manager is None:
        with model_init_lock:
            if model_manager is None:
                if model_config_path is None:
                    raise RuntimeError("model_config_path not set. Server not properly initialized.")
                print(f"Initializing model manager with config: {model_config_path}")
                model_manager = get_model_manager(
                    config_path=model_config_path,
                    traj_mask_cfg=traj_mask_cfg,
                )
                print("Model manager ready!")
    return model_manager


def _load_first_caption(text_path: str) -> str:
    with open(text_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return line.split("#")[0].strip()
    return ""


def _resample_uniform_arclength(points_xyz: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a world-space XZ polyline to uniformly spaced points."""
    points = np.asarray(points_xyz, dtype=np.float32)
    if num_points <= 0 or len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if num_points == 1 or len(points) == 1:
        return points[:1].astype(np.float32)

    seg_lens = np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1)
    total_len = float(seg_lens.sum())
    if total_len <= 1e-6:
        return np.repeat(points[:1].astype(np.float32), num_points, axis=0)
    return resample_polyline(
        points,
        num_tokens=num_points,
        token_step=total_len / float(num_points - 1),
    )


def load_debug_preset_sample():
    """Load a HumanML3D root trajectory preset for web-demo sanity checks.

    The preset intentionally passes only world-space root points to the normal
    web trajectory path.  ModelManager then assigns timestamps with
    traj_mask.waypoint_dt, matching user-drawn paths instead of using a separate
    debug-only timestamp source.
    """
    cfg = debug_preset_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return None

    dataset = str(cfg.get("dataset", "humanml3d")).lower()
    sample_id = str(cfg.get("sample_id", "001168"))
    raw_data_dir = cfg.get("raw_data_dir")
    if not raw_data_dir:
        raise ValueError("web_demo_debug.raw_data_dir is required when debug preset is enabled")

    if dataset != "humanml3d":
        raise ValueError(f"Unsupported web_demo_debug.dataset: {dataset}")

    data_dir = os.path.join(raw_data_dir, "HumanML3D")
    feature_path = os.path.join(
        data_dir,
        str(cfg.get("feature_path", "new_joint_vecs")),
        f"{sample_id}.npy",
    )
    text_path = os.path.join(
        data_dir,
        str(cfg.get("text_path", "texts")),
        f"{sample_id}.txt",
    )
    feature = np.load(feature_path).astype(np.float32)
    root = extract_root_trajectory_263(feature).astype(np.float32)
    root = _resample_uniform_arclength(root, len(root))
    waypoint_dt = float((traj_mask_cfg or {}).get("waypoint_dt", 0.05))
    text = str(cfg.get("text", "")).strip() or _load_first_caption(text_path)
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "text": text,
        "trajectory": root,
        "num_frames": int(len(feature)),
        "duration_seconds": float(max(0, len(root) - 1) * waypoint_dt),
        "waypoint_dt": waypoint_dt,
    }


@app.after_request
def add_no_cache_headers(response):
    """Avoid stale JS/CSS while iterating on the web demo."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _handle_consumption_timeout(session_id: str, elapsed: float) -> bool:
    """Reset generation when the active client stops consuming frames."""
    if not model_manager or not model_manager.is_generating:
        return False
    print(
        f"No frame consumed for {elapsed:.1f}s - client disconnected, auto-resetting..."
    )
    if not model_manager.reset():
        print("Auto-reset skipped because the generation thread did not stop cleanly")
        return False
    print("Generation reset due to client disconnect (no frame consumption)")
    return True


def start_consumption_monitor():
    """Start the consumption monitoring thread if not already running."""
    session_service.start_consumption_monitor(_handle_consumption_timeout)
    print("Consumption monitor started")


def get_trajectory_defaults() -> dict:
    traj_cfg = traj_mask_cfg or {}
    root_cfg = (traj_cfg.get("root_refiner", {}) or {})
    route_mode = str(root_cfg.get("route_mode", "relative_to_actor"))
    return {
        "route_mode": route_mode,
        "horizon_tokens": int(traj_cfg.get("horizon_tokens", 20)),
        "delay_enabled": bool(traj_cfg.get("update_delay_enabled", True)),
        "delay_tokens": int(
            traj_cfg.get("update_delay_tokens", traj_cfg.get("horizon_tokens", 20))
        ),
    }


register_routes(
    app,
    init_model=init_model,
    get_model_manager=lambda: model_manager,
    load_debug_preset_sample=load_debug_preset_sample,
    start_consumption_monitor=start_consumption_monitor,
    session_service=session_service,
    get_trajectory_defaults=get_trajectory_defaults,
)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Flask server for real-time 3D motion generation')
    parser.add_argument('--config', type=str, default='../configs/stream.yaml',
                        help='Path to config yaml file (default: ../configs/stream.yaml)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    args = parser.parse_args()

    model_config_path = args.config
    traj_mask_cfg = load_traj_mask_cfg(model_config_path)
    debug_preset_cfg = load_debug_preset_cfg(model_config_path)
    
    print("Starting Flask server...")
    print(f"Config file: {model_config_path}")
    print("Trajectory config source: traj_mask section in main config")
    if debug_preset_cfg and bool(debug_preset_cfg.get("enabled", False)):
        print(f"Web demo debug preset enabled: {debug_preset_cfg}")
    print("Note: Model will be loaded on first generation request")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
