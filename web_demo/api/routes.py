"""HTTP route registration for the FloodNet web demo."""

from __future__ import annotations

from flask import jsonify, render_template, request

from web_demo.api.schemas import UpdateTrajectoryRequest


def register_routes(
    app,
    *,
    init_model,
    get_model_manager,
    load_debug_preset_sample,
    start_consumption_monitor,
    session_service,
    get_trajectory_defaults,
):
    """Register all web demo routes without owning runtime state."""

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            trajectory_defaults=get_trajectory_defaults(),
        )

    @app.route("/api/start", methods=["POST"])
    def start_generation():
        try:
            session_claimed = False

            data = request.get_json(silent=True) or {}
            session_id = data.get("session_id")
            text = data.get("text", "walk in a circle.")
            history_length = data.get("history_length", 30)
            smoothing_alpha = data.get("smoothing_alpha", None)
            denoise_steps = data.get("denoise_steps", None)
            root_feedback_enabled = data.get("root_feedback_enabled", None)
            root_feedback_xz_blend_alpha = data.get(
                "root_feedback_xz_blend_alpha",
                None,
            )
            force = data.get("force", False)

            if not session_id:
                return jsonify({
                    "status": "error",
                    "message": "session_id is required",
                }), 400

            print(
                f"[Session {session_id}] Starting generation with text: {text}, "
                f"history_length: {history_length}, force: {force}"
            )

            mm = init_model()
            debug_sample = load_debug_preset_sample()
            if debug_sample is not None:
                text = debug_sample["text"]
                print(
                    f"[Session {session_id}] web_demo_debug enabled: "
                    f"{debug_sample['dataset']}:{debug_sample['sample_id']} "
                    f"frames={debug_sample['num_frames']} "
                    f"duration={debug_sample['duration_seconds']:.2f}s",
                    flush=True,
                )

            claim = session_service.claim_generation(
                session_id,
                force=force,
                is_generating=mm.is_generating,
            )
            if not claim.ok:
                payload = {
                    "status": "error",
                    "message": claim.message,
                }
                if claim.conflict:
                    payload["conflict"] = True
                    payload["active_session_id"] = claim.active_session_id
                return jsonify(payload), claim.status_code
            session_claimed = True

            if claim.need_force_takeover:
                print(
                    f"[Session {session_id}] Force takeover from "
                    f"session {claim.previous_session_id}"
                )
                session_service.clear_consumption()

            if mm.is_generating:
                if not mm.pause_generation():
                    session_service.release(session_id)
                    return jsonify({
                        "status": "error",
                        "message": (
                            "Previous generation did not stop cleanly. "
                            "Please retry reset/start."
                        ),
                    }), 503

            if not mm.reset(
                history_length=history_length,
                smoothing_alpha=smoothing_alpha,
                denoise_steps=denoise_steps,
                root_feedback_enabled=root_feedback_enabled,
                root_feedback_xz_blend_alpha=root_feedback_xz_blend_alpha,
            ):
                session_service.release(session_id)
                return jsonify({
                    "status": "error",
                    "message": (
                        "Model reset failed because the previous generation "
                        "thread is still alive."
                    ),
                }), 503

            debug_target_traj = None
            if debug_sample is not None:
                mm.configure_debug_repeat(
                    debug_sample["trajectory"],
                    debug_sample.get("repeat", {}),
                    duration_seconds=debug_sample.get("duration_seconds"),
                )
                debug_target_traj = mm.update_trajectory(
                    debug_sample["trajectory"],
                    mode="replace_future",
                    source="debug_preset",
                    duration_seconds=debug_sample.get("duration_seconds"),
                )
            mm.start_generation(text, history_length=history_length)

            session_service.touch_consumption()
            start_consumption_monitor()
            print(f"[Session {session_id}] Consumption monitoring activated")

            return jsonify({
                "status": "success",
                "message": (
                    f"Generation started with text: {text}, "
                    f"history_length: {history_length}"
                ),
                "session_id": session_id,
                "text": text,
                "debug_preset": None if debug_sample is None else {
                    "dataset": debug_sample["dataset"],
                    "sample_id": debug_sample["sample_id"],
                    "num_frames": debug_sample["num_frames"],
                    "duration_seconds": debug_sample["duration_seconds"],
                },
                "trajectory": (
                    None
                    if debug_target_traj is None
                    else debug_target_traj.tolist()
                ),
            })
        except Exception as exc:
            if "session_id" in locals() and session_claimed:
                session_service.release(session_id)
                session_service.clear_consumption()
            print(f"Error in start_generation: {exc}")
            import traceback

            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    @app.route("/api/update_text", methods=["POST"])
    def update_text():
        try:
            data = request.get_json(silent=True) or {}
            session_id = data.get("session_id")
            text = data.get("text", "")

            if not session_id:
                return jsonify({
                    "status": "error",
                    "message": "session_id is required",
                }), 400

            if not session_service.is_active(session_id):
                return jsonify({
                    "status": "error",
                    "message": "Not the active session",
                }), 403

            model_manager = get_model_manager()
            if model_manager is None:
                return jsonify({
                    "status": "error",
                    "message": "Model not initialized",
                }), 400

            model_manager.update_text(text)
            return jsonify({
                "status": "success",
                "message": f"Text updated to: {text}",
            })
        except Exception as exc:
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    @app.route("/api/update_trajectory", methods=["POST"])
    def update_trajectory():
        try:
            data = request.get_json(silent=True) or {}
            req = UpdateTrajectoryRequest.from_payload(data)

            if not req.session_id:
                return jsonify({
                    "status": "error",
                    "message": "session_id is required",
                }), 400

            model_manager = get_model_manager()
            if model_manager is None:
                return jsonify({
                    "status": "error",
                    "message": "Model not initialized (start generation first)",
                }), 400

            if not session_service.is_active(req.session_id):
                return jsonify({
                    "status": "error",
                    "message": "Not the active session",
                }), 403

            target_traj = model_manager.update_trajectory(
                req.waypoints,
                mode=req.mode,
                source=req.source,
                duration_seconds=req.duration_seconds,
                route_mode=req.route_mode,
                horizon_tokens=req.horizon_tokens,
                delay_enabled=req.delay_enabled,
                delay_tokens=req.delay_tokens,
            )
            target_len = 0 if target_traj is None else len(target_traj)
            print(
                f"[Session {req.session_id}] update_trajectory mode={req.mode} "
                f"waypoints={0 if not req.waypoints else len(req.waypoints)} "
                f"target_len={target_len}",
                flush=True,
            )

            return jsonify({
                "status": "success",
                "message": (
                    "Trajectory updated" if req.waypoints else "Trajectory cleared"
                ),
                "mode": req.mode,
                "route_mode": getattr(model_manager, "route_reference_mode", None),
                "horizon_tokens": getattr(model_manager, "traj_horizon_tokens", None),
                "delay_enabled": getattr(
                    model_manager,
                    "traj_update_delay_enabled",
                    None,
                ),
                "delay_tokens": getattr(
                    model_manager,
                    "traj_update_delay_tokens",
                    None,
                ),
                "trajectory": (
                    target_traj.tolist() if target_traj is not None else None
                ),
            })
        except Exception as exc:
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    @app.route("/api/pause", methods=["POST"])
    def pause_generation():
        try:
            data = request.get_json(silent=True) or {}
            session_id = data.get("session_id")

            if not session_id:
                return jsonify({
                    "status": "error",
                    "message": "session_id is required",
                }), 400

            if not session_service.is_active(session_id):
                return jsonify({
                    "status": "error",
                    "message": "Not the active session",
                }), 403

            model_manager = get_model_manager()
            if model_manager:
                model_manager.pause_generation()

            return jsonify({
                "status": "success",
                "message": "Generation paused",
            })
        except Exception as exc:
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    @app.route("/api/resume", methods=["POST"])
    def resume_generation():
        try:
            data = request.get_json(silent=True) or {}
            session_id = data.get("session_id")

            if not session_id:
                return jsonify({
                    "status": "error",
                    "message": "session_id is required",
                }), 400

            if not session_service.is_active(session_id):
                return jsonify({
                    "status": "error",
                    "message": "Not the active session",
                }), 403

            model_manager = get_model_manager()
            if model_manager is None:
                return jsonify({
                    "status": "error",
                    "message": "Model not initialized",
                }), 400

            model_manager.resume_generation()
            session_service.touch_consumption()

            return jsonify({
                "status": "success",
                "message": "Generation resumed",
            })
        except Exception as exc:
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    @app.route("/api/reset", methods=["POST"])
    def reset():
        try:
            data = request.get_json(silent=True) or {}
            session_id = data.get("session_id")
            history_length = data.get("history_length", 30)
            smoothing_alpha = data.get("smoothing_alpha", None)
            denoise_steps = data.get("denoise_steps", None)
            root_feedback_enabled = data.get("root_feedback_enabled", None)
            root_feedback_xz_blend_alpha = data.get(
                "root_feedback_xz_blend_alpha",
                None,
            )

            if session_id and not session_service.can_reset(session_id):
                return jsonify({
                    "status": "error",
                    "message": "Not the active session",
                }), 403

            model_manager = get_model_manager()
            if model_manager:
                if not model_manager.reset(
                    history_length=history_length,
                    smoothing_alpha=smoothing_alpha,
                    denoise_steps=denoise_steps,
                    root_feedback_enabled=root_feedback_enabled,
                    root_feedback_xz_blend_alpha=root_feedback_xz_blend_alpha,
                ):
                    return jsonify({
                        "status": "error",
                        "message": (
                            "Model reset failed because the previous generation "
                            "thread is still alive."
                        ),
                    }), 503

            session_service.release(session_id or None)
            session_service.clear_consumption()

            if model_manager and model_manager.is_generating:
                if not model_manager.pause_generation():
                    return jsonify({
                        "status": "error",
                        "message": (
                            "Generation thread did not stop cleanly after reset."
                        ),
                    }), 503

            params_msg = (
                f", smoothing: {smoothing_alpha}"
                if smoothing_alpha is not None
                else ""
            )
            params_msg += (
                f", steps: {denoise_steps}" if denoise_steps is not None else ""
            )
            print(f"[Session {session_id}] Reset complete, session cleared")

            return jsonify({
                "status": "success",
                "message": (
                    f"Reset complete with history_length: "
                    f"{history_length}{params_msg}"
                ),
            })
        except Exception as exc:
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    @app.route("/api/get_frame", methods=["GET"])
    def get_frame():
        try:
            session_id = request.args.get("session_id")

            if not session_id:
                return jsonify({
                    "status": "error",
                    "message": "session_id is required",
                }), 400

            if not session_service.is_active(session_id):
                return jsonify({
                    "status": "error",
                    "message": "Not the active session",
                }), 403

            model_manager = get_model_manager()
            if model_manager is None:
                return jsonify({
                    "status": "error",
                    "message": "Model not initialized",
                }), 400

            joints, traj = model_manager.get_next_frame()

            if joints is not None:
                session_service.touch_consumption()
                response = {
                    "status": "success",
                    "joints": joints.tolist(),
                    "buffer_size": model_manager.frame_buffer.size(),
                }
                if traj is not None:
                    response["trajectory"] = traj.tolist()
                snapshot_revision = request.args.get(
                    "trajectory_snapshot_revision"
                )
                try:
                    snapshot_revision = (
                        None
                        if snapshot_revision is None
                        else int(snapshot_revision)
                    )
                except ValueError:
                    snapshot_revision = None
                response["trajectory_debug"] = model_manager.get_trajectory_debug(
                    client_snapshot_revision=snapshot_revision,
                )
                return jsonify(response)
            return jsonify({
                "status": "waiting",
                "message": "No frame available yet",
                "buffer_size": model_manager.frame_buffer.size(),
            })
        except Exception as exc:
            print(f"Error in get_frame: {exc}")
            import traceback

            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    @app.route("/api/status", methods=["GET"])
    def get_status():
        try:
            session_id = request.args.get("session_id")
            is_active_session, active_session_id = session_service.active_status(
                session_id
            )

            model_manager = get_model_manager()
            if model_manager is None:
                return jsonify({
                    "initialized": False,
                    "buffer_size": 0,
                    "is_generating": False,
                    "is_active_session": is_active_session,
                    "active_session_id": active_session_id,
                })

            status = model_manager.get_buffer_status()
            status["initialized"] = True
            status["is_active_session"] = is_active_session
            status["active_session_id"] = active_session_id
            return jsonify(status)
        except Exception as exc:
            return jsonify({
                "status": "error",
                "message": str(exc),
            }), 500

    return app


__all__ = ["register_routes"]
