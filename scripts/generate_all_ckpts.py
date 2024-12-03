"""
Use `repa` environment
"""
import argparse
import os
import json
import subprocess


def main(args):
    command_list_base = [
        "torchrun", "--nnodes=1", f"--nproc_per_node={args.n_gpus}", "generate.py",
        "--num-fid-samples", "50000",
        "--path-type", "linear",
        "--per-proc-batch-size", str(args.per_gpu_batch_size),
        "--encoder-depth", str(args.enc_dep),
        "--projector-embed-dims", str(args.z_dim),
        "--mode", "sde",
        "--num-steps", "250",
        "--cfg-scale", str(args.cfg_scale),
        "--guidance-high", str(args.guidance_high),
        "--guidance-low", str(args.guidance_low),
    ]
    if args.ckpt_path is None:
        try:
            result = subprocess.run(
                command_list_base,
                check=True,
                text=True,
            )
            print("Command executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Command failed.")
            print("Exit code:", e.returncode)

    assert os.path.exists(os.path.join(args.ckpt_path, "checkpoints")), "No checkpoints found in the specified directory."
    ckpt_files = sorted(os.listdir(os.path.join(args.ckpt_path, "checkpoints")))

    # Load config:
    with open(os.path.join(args.ckpt_path, "args.json"), "r") as f:
        config = json.load(f)

    # Generate all checkpoints by forking subprocesses
    for ckpt_file in ckpt_files:
        command_list = command_list_base + [
            "--ckpt", os.path.join(args.ckpt_path, "checkpoints", ckpt_file),
            "--model", config["model"],
        ]
        try:
            result = subprocess.run(
                command_list,
                check=True,
                text=True,
            )
            print("Command executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Command failed.")
            print("Exit code:", e.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--cfg-scale', type=float, default=1.0)  # <- By default turning off the cfg
    parser.add_argument('--guidance-low', type=float, default=0.0)
    parser.add_argument('--guidance-high', type=float, default=1.0)
    parser.add_argument('--enc_dep', type=int, default=8)
    parser.add_argument('--z_dim', type=str, default="768")
    parser.add_argument('--n_gpus', type=int, default=4)
    parser.add_argument('--per_gpu_batch_size', type=int, default=8)
    args = parser.parse_args()
    main(args)
