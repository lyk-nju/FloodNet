import argparse
from collections import OrderedDict

import torch


def rename_checkpoint_keys(ckpt_path, save_path, old_prefix, new_prefix):
    """
    Rename keys in checkpoint by replacing prefix

    Args:
        ckpt_path: path to input checkpoint
        save_path: path to save modified checkpoint
        old_prefix: prefix to remove/replace
        new_prefix: new prefix to add (can be empty string)
    """
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Process state_dict
    if "state_dict" in checkpoint:
        old_state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()

        for old_key, value in old_state_dict.items():
            if old_key.startswith(old_prefix):
                new_key = old_key[len(old_prefix) :]  # Remove old prefix
                if new_prefix:
                    new_key = new_prefix + new_key  # Add new prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[old_key] = value

        checkpoint["state_dict"] = new_state_dict
        print(
            f"Renamed {len([k for k in old_state_dict.keys() if k.startswith(old_prefix)])} keys"
        )
        print(f"Old prefix: '{old_prefix}' -> New prefix: '{new_prefix}'")

    # Save modified checkpoint
    torch.save(checkpoint, save_path)
    print(f"Saved modified checkpoint to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename keys in PyTorch checkpoint")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Input checkpoint path"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Output checkpoint path"
    )
    parser.add_argument(
        "--old_prefix", type=str, required=True, help="Prefix to remove/replace"
    )
    parser.add_argument(
        "--new_prefix", type=str, default="", help="New prefix to add (default: empty)"
    )

    args = parser.parse_args()

    rename_checkpoint_keys(
        ckpt_path=args.ckpt_path,
        save_path=args.save_path,
        old_prefix=args.old_prefix,
        new_prefix=args.new_prefix,
    )
