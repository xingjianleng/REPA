# Reproducing Table 2 in the paper, last line
accelerate launch train.py \
    --gradient-accumulation-steps=2 \
    --max-train-steps=400000 \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-L/2" \
    --enc-type="dinov2-vit-l" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --output-dir="exps" \
    --exp-name="sit-l-linear-dinov2-l-enc8-400k" \
    --data-dir=data/
