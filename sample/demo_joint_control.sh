#!/usr/bin/env bash
set -euo pipefail

# Generate each arbitrary-frame/arbitrary-joint sample sequentially on one GPU.
# Usage: bash sample/demo_joint_control.sh [gpu] [num_steps]

readonly NAME="MMDiT_xyz_pretrained"
readonly GPU="${1:-0}"
readonly SPEC_DIR="sample/joint_control_specs"
readonly -a LABELS=(kick walk cartwheel run_circle)

for label in "${LABELS[@]}"; do
    out_dir="generations/joint_control/${label}"
    echo "[GPU ${GPU}] ${label} -> ${out_dir}"

    uv run python sample/demo_joint_contorl.py \
        --name "${NAME}" \
        --ckpt latest \
        --gpu "${GPU}" \
        --sparse_control_spec "${SPEC_DIR}/${label}.json" \
        --out_dir "${out_dir}" \
        --num_samples 1 \
        --num_steps 500 \
        --cfg 2 \
        --w_kin 1 \
        --use_metric_R true \
        --use_augmented_obs true \
        --use_noise_mixing true \
        --noise_mix_strength 1 \
        --save_mp4
done

