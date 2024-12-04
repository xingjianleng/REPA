"""
Use the `diffusion_eval` environment for evaluating all npz files
"""
import argparse
import re
import pandas as pd
import os
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
    assert os.path.exists(os.path.join(args.guided_diffusion_eval_path, "classify_image_graph_def.pb"))
    assert os.path.exists(args.npz_path)

    output_dir = f"{args.output_path}/{os.path.basename(args.npz_path).replace('.npz', '.csv')}"

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    command_list = [
        "python", os.path.join(args.guided_diffusion_eval_path, "evaluator.py"),
        os.path.join(args.guided_diffusion_eval_path, "VIRTUAL_imagenet256_labeled.npz"), args.npz_path
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
    parser.add_argument('--npz-path', type=str, default="")
    parser.add_argument('--output-path', type=str, default='evals')
    args = parser.parse_args()
    main(args)
