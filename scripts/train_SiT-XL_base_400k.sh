# Reproducing Table 3 in the paper, the base SiT-XL/2 model
accelerate launch train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/2" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.0 \
    --encoder-depth=8 \
    --output-dir="exps" \
    --exp-name="sit-xl-base-400k" \
    --data-dir=data/