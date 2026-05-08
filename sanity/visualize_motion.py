import os

from utils.visualize import render_video

RAW_DATA_PATH = "raw_data"
feature_dirs = [
    f"{RAW_DATA_PATH}/BABEL_streamed/motions",
    f"{RAW_DATA_PATH}/BABEL_streamed/motions_recovered_20251030_085836_vae_wan_z4",
    f"{RAW_DATA_PATH}/HumanML3D/new_joint_vecs",
    f"{RAW_DATA_PATH}/HumanML3D/new_joint_vecs_recovered_TOKENS_20251030_085836_vae_wan_z4",
]
save_dirs = [
    f"{RAW_DATA_PATH}/BABEL_streamed/animations",
    f"{RAW_DATA_PATH}/BABEL_streamed/animations_recovered_20251030_085836_vae_wan_z4",
    f"{RAW_DATA_PATH}/HumanML3D/animations",
    f"{RAW_DATA_PATH}/HumanML3D/animations_recovered_20251030_085836_vae_wan_z4",
]
frame_dirs = [
    f"{RAW_DATA_PATH}/BABEL_streamed/frames",
    f"{RAW_DATA_PATH}/BABEL_streamed/frames",
    None,
    None,
]

# make dir
for save_dir in save_dirs:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
render_setting = {}
render_setting["recover_dim"] = 263
render_setting["simple"] = True

for feature_dir, save_dir, frame_dir in zip(feature_dirs, save_dirs, frame_dirs):
    print(f"Rendering {feature_dir} to {save_dir}")
    render_video(
        motion_dir=feature_dir,
        save_dir=save_dir,
        render_setting=render_setting,
        frames_dir=frame_dir,
    )
