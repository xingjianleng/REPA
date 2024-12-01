# Reproducing Table 3 in the paper, the SiT model + REPA
accelerate launch train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-B/2" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --output-dir="exps" \
    --exp-name="sit-b-linear-dinov2-b-enc8-400k" \
    --data-dir=data/
