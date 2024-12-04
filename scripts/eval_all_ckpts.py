"""
Use the `diffusion_eval` environment for evaluating all npz files
"""
import argparse
import re
import pandas as pd
import os
from tqdm import tqdm
import json
import subprocess


METRICS = {
    "Inception Score": r"Inception Score:\s*([\d\.]+)",
    "FID": r"FID:\s*([\d\.]+)",
    "sFID": r"sFID:\s*([\d\.]+)",
    "Precision": r"Precision:\s*([\d\.]+)",
    "Recall": r"Recall:\s*([\d\.]+)"
}


def main(args):
    # Base files used in evaluation
    assert os.path.exists(os.path.join(args.guided_diffusion_eval_path, "VIRTUAL_imagenet256_labeled.npz"))
    assert os.path.exists(os.path.join(args.guided_diffusion_eval_path, "evaluator.py"))
    # assert os.path.exists(os.path.join(args.guided_diffusion_eval_path, "classify_image_graph_def.pb"))

    sample_folder_names = []
    output_dirs = []
    if args.ckpt_path is None:
        sample_folder_names.append(
            f"{args.sample_dir}/SiT-XL-2-SiT-XL-2-256x256-size-256-vae-ema-cfg-1.0-0.0-1.0-seed-0-sde.npz"
        )
        output_dirs.append(
            f"{args.output_path}/default_example/size-256-vae-ema-cfg-1.0-0.0-1.0-seed-0-sde/last.csv"
        )

    else:
        assert os.path.exists(os.path.join(args.ckpt_path, "checkpoints")), "No checkpoints found in the specified directory."
        for ckpt_file in sorted(os.listdir(os.path.join(args.ckpt_path, "checkpoints"))):
            with open(os.path.join(args.ckpt_path, "args.json"), "r") as f:
                config = json.load(f)
            model = config["model"]
            model_string_name = model.replace("/", "-")
            exp_name = os.path.basename(args.ckpt_path)
            ckpt_string_name = os.path.basename(ckpt_file).replace(".pt", "")
            folder_name = f"{exp_name}{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-{args.vae}-" \
                        f"cfg-{args.cfg_scale}-{args.guidance_low}-{args.guidance_high}-seed-{args.global_seed}-{args.mode}"
            sample_folder_names.append(f"{args.sample_dir}/{folder_name}.npz")
            output_dirs.append(
                f"{args.output_path}/{exp_name}/size-{args.resolution}-vae-{args.vae}-" \
                f"cfg-{args.cfg_scale}-{args.guidance_low}-{args.guidance_high}-seed-{args.global_seed}-{args.mode}/{ckpt_string_name}.csv")

    for sample_folder_name, output_dir in tqdm(zip(sample_folder_names, output_dirs), total=len(sample_folder_names)):
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        command_list = [
            "python", os.path.join(args.guided_diffusion_eval_path, "evaluator.py"),
            os.path.join(args.guided_diffusion_eval_path, "VIRTUAL_imagenet256_labeled.npz"), sample_folder_name
        ]
        try:
            result = subprocess.run(
                command_list,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("Command executed successfully.")
        except subprocess.CalledProcessError as e:
            print("Command failed.")
            print("Exit code:", e.returncode)

        extracted_metrics = {}
        for key, pattern in METRICS.items():
            match = re.search(pattern, result.stdout)
            if match:
                extracted_metrics[key] = float(match.group(1))
            else:
                extracted_metrics[key] = None

        df = pd.DataFrame([extracted_metrics])
        df.to_csv(output_dir, index=False)


if __name__ == "__main__":
    # The argparse helps to control which model and which hyperparameters to evaluate
    parser = argparse.ArgumentParser()
    parser.add_argument('--guided-diffusion-eval-path', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--sample-dir', type=str, default='samples')
    parser.add_argument('--output-path', type=str, default='evals')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--vae', type=str, default='ema')
    parser.add_argument('--cfg-scale', type=float, default=1.0)
    parser.add_argument('--guidance-low', type=float, default=0.0)
    parser.add_argument('--guidance-high', type=float, default=1.0)
    parser.add_argument('--global-seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='sde')
    args = parser.parse_args()
    main(args)
