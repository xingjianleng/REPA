import os
import time
import yaml
import subprocess

# Constants
EXP_FOLDER = "exps/"
EVAL_FOLDER = "evals_2/"
SAMPLES_FOLDER = "samples/"
JOB_FILES_TEMP = "job_files_temp/"
CFG_PATH = "configs/eval_monitor_cfg.yaml"
SCAN_INTERVAL = 5

# SLURM job script template
TEMPLATE = """#!/bin/bash
#SBATCH --account=OD-227441
#SBATCH --job-name=eval_monitor_generated_job
#SBATCH --output=slurm_outputs/eval_monitor_generated_job_%A_%j.out
#SBATCH --error=slurm_outputs/eval_monitor_generated_job_%A_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2

source /scratch3/zha439/miniconda3/bin/activate

conda activate repa
torchrun --nnodes=1 --nproc_per_node=2 --master_port=7234 generate.py \
    --num-fid-samples 20000 \
    --path-type linear \
    --per-proc-batch-size 256 \
    --encoder-depth 8 \
    --projector-embed-dims {projector_embed_dims} \
    --mode sde \
    --num-steps 250 \
    --cfg-scale 1.0 \
    --guidance-high 1.0 \
    --guidance-low 0.0 \
    --sample-dir {samples_folder} \
    --ckpt {ckpt} \
    --exp-name {exp_name} \
    --model {model}

conda activate diffusion_eval
python scripts/eval_single.py --guided-diffusion-eval-path /scratch3/zha439/guided-diffusion/evaluations/ --npz-path {npz_path} --output-path {eval_folder}

rm {npz_path}
rm -r {exp_sample_dir}"""


def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('monitored_experiments', {})


def main():
    # Ensure necessary directories exist
    os.makedirs(EVAL_FOLDER, exist_ok=True)
    os.makedirs(SAMPLES_FOLDER, exist_ok=True)
    os.makedirs(JOB_FILES_TEMP, exist_ok=True)

    # Load monitored experiments from the configuration file
    monitored_experiments = load_config(CFG_PATH)
    if not monitored_experiments:
        print("No experiments to monitor. Exiting.")
        return

    while True:
        for exp_name, params in monitored_experiments.items():
            ckpt_path = os.path.join(EXP_FOLDER, exp_name, "checkpoints")
            if not os.path.exists(ckpt_path):
                continue

            # List all checkpoint files
            ckpt_files = [f for f in os.listdir(ckpt_path) if f.endswith(".pt")]
            for ckpt_file in ckpt_files:
                exp_name_ckpt = f"{exp_name}_{ckpt_file.replace('.pt', '')}"
                if os.path.exists(os.path.join(EVAL_FOLDER, f"{exp_name_ckpt}.csv")):
                    continue

                ckpt_full_path = os.path.join(ckpt_path, ckpt_file)
                print(f"New checkpoint detected: {ckpt_full_path}")

                # Prepare SLURM job script content
                job_script_content = TEMPLATE.format(
                    projector_embed_dims=params['projector_embed_dims'],
                    ckpt=ckpt_full_path,
                    exp_name=exp_name_ckpt,
                    samples_folder=SAMPLES_FOLDER,
                    model=params['model'],
                    npz_path=os.path.join(SAMPLES_FOLDER, f"{exp_name_ckpt}.npz"),
                    eval_folder=EVAL_FOLDER,
                    exp_sample_dir=os.path.join(SAMPLES_FOLDER, exp_name),
                )

                # Write the job script to a temporary file
                job_script_path = os.path.join(JOB_FILES_TEMP, f"eval_monitor_generated_job_{exp_name_ckpt}.sh")
                with open(job_script_path, 'w') as job_file:
                    job_file.write(job_script_content)

                # Submit the SLURM job
                try:
                    result = subprocess.run(['sbatch', job_script_path], check=True, capture_output=True, text=True)
                    print(f"Submitted SLURM job for {ckpt_file}: {result.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to submit SLURM job for {ckpt_file}: {e.stderr.strip()}")

        # Wait before the next scan
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
