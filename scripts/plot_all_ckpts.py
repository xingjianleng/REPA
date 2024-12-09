"""
Use this to plot FID scores
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


EXPS = {
    "sit-b-base-400k-SiT-B-2-": "sit-b-base-400k",
    "sit-b-linear-dinov2-b-enc8-400k-SiT-B-2-": "sit-b-linear-dinov2-b-enc8-400k",
    # "sit-l-base-400k-SiT-L-2-": "sit-l-base-400k",
    # "sit-l-linear-dinov2-l-enc8-400k-SiT-L-2-": "sit-l-linear-dinov2-l-enc8-400k",
    # "sit-xl-base-400k-SiT-XL-2-": "sit-xl-base-400k",
    # "sit-xl-linear-dinov2-b-enc8-400k-SiT-XL-2-": "sit-xl-linear-dinov2-b-enc8-400k",
    "sit-b-linear-dinov2-b-enc8-patch-0.5-400k-SiT-B-2-": "sit-b-linear-dinov2-b-enc8-patch-0.5-400k",
    "sit-b-linear-dinov2-b-enc8-patch-0.75-400k-SiT-B-2-": "sit-b-linear-dinov2-b-enc8-patch-0.75-400k",
    "sit-b-linear-dinov2-b-enc8-repa-patch-0.5-400k-SiT-B-2-": "sit-b-linear-dinov2-b-enc8-repa-patch-0.5-400k",
    "sit-b-linear-dinov2-b-enc8-repa-patch-0.75-400k-SiT-B-2-": "sit-b-linear-dinov2-b-enc8-repa-patch-0.75-400k",
    "sit-b-linear-dinov2-b-enc8-repa-patch-1.0-400k-SiT-B-2-": "sit-b-linear-dinov2-b-enc8-repa-patch-1.0-400k",
}


def main(args):
    # Choose the HP we want to eval
    eval_cfg = f"-size-{args.resolution}-vae-{args.vae}-" \
               f"cfg-{args.cfg_scale}-{args.guidance_low}-{args.guidance_high}-seed-{args.global_seed}-{args.mode}"
    csv_files = sorted([f for f in os.listdir(args.eval_dir) if f.endswith(".csv") and eval_cfg in f])

    # Load FID scores
    fid_scores_steps_exps = {}
    for k, v in EXPS.items():
        steps = []
        fid_scores = []

        chosen_csv_files = [f for f in csv_files if k in f]
        for csv_file in chosen_csv_files:
            df = pd.read_csv(os.path.join(args.eval_dir, csv_file))
            fid = df["FID"].values.item()
            step = int(csv_file.split(k)[1].split(eval_cfg)[0])
            fid_scores.append(fid)
            steps.append(step)

        fid_scores_steps_exps[v] = (steps, fid_scores)
    
    # Plot FID scores
    plt.figure(figsize=(8, 6))
    for exp, (steps, fid_scores) in fid_scores_steps_exps.items():
        plt.plot(steps, fid_scores, label=exp, marker='o')
    plt.xlabel("Steps")
    plt.ylabel("FID")
    plt.legend()
    plt.grid()
    plt.title(f"FID scores {eval_cfg}")
    plt.savefig(f"{args.eval_dir}/{eval_cfg}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', type=str, default='evals')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--vae', type=str, default='ema')
    parser.add_argument('--cfg-scale', type=float, default=1.0)
    parser.add_argument('--guidance-low', type=float, default=0.0)
    parser.add_argument('--guidance-high', type=float, default=1.0)
    parser.add_argument('--global-seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='sde')
    args = parser.parse_args()
    main(args)
