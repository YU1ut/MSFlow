"""Evaluation loop for direct SnapMoGen motion generation."""

import numpy as np
import torch
from tqdm import tqdm

from utils.snap_metrics import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
    calculate_R_precision,
    cosine_similarity_matrix,
)


def _as_motion_features(motions):
    if motions.ndim == 4:
        return motions[:, :, 0]
    if motions.ndim == 3:
        return motions
    raise ValueError(f"Expected [B, T, D] or [B, T, 1, D], got {motions.shape}")


@torch.no_grad()
def evaluation_msflow_snap(
    out_dir,
    val_loader,
    ema_model,
    ae,
    ep,
    best_fid,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    device,
    clip_score_old,
    cond_scale=None,
    cal_mm=False,
    eval_mean=None,
    eval_std=None,
    after_mean=None,
    after_std=None,
    return_current=False,
    **unused_kwargs,
):
    del out_dir, ae, eval_mean, eval_std, after_mean, after_std, unused_kwargs
    ema_model.eval()
    cond_scale = 3.0 if cond_scale is None else float(cond_scale)

    ground_truth_embeddings = []
    generated_embeddings = []
    multimodality_embeddings = []
    r_precision_real = np.zeros(3)
    r_precision = np.zeros(3)
    matching_real = 0.0
    matching = 0.0
    sample_count = 0
    multimodality_batches = 3 if cal_mm else 0

    for batch_index, batch in enumerate(tqdm(val_loader)):
        captions, motions, motion_lengths = batch
        motions = motions.to(device).float()
        motion_lengths = motion_lengths.to(device).long()
        motion_features = _as_motion_features(motions)

        text_embeddings = eval_wrapper.encode_text(captions)
        ground_truth_fid, ground_truth_text_space = eval_wrapper.encode_motion(
            motion_features, motion_lengths
        )

        generation_count = 30 if batch_index < multimodality_batches else 1
        batch_embeddings = []
        generated_fid = None
        generated_text_space = None
        for _ in range(generation_count):
            generated = ema_model.generate(
                captions,
                motion_lengths,
                cond_scale=cond_scale,
            )
            generated_fid, generated_text_space = eval_wrapper.encode_motion(
                _as_motion_features(generated), motion_lengths
            )
            batch_embeddings.append(generated_text_space.unsqueeze(1))

        if generation_count > 1:
            multimodality_embeddings.append(torch.cat(batch_embeddings, dim=1))

        assert generated_fid is not None and generated_text_space is not None
        ground_truth_embeddings.append(ground_truth_fid)
        generated_embeddings.append(generated_fid)

        text_np = text_embeddings.cpu().numpy()
        ground_truth_np = ground_truth_text_space.cpu().numpy()
        generated_np = generated_text_space.cpu().numpy()
        r_precision_real += calculate_R_precision(
            text_np,
            ground_truth_np,
            top_k=3,
            sum_all=True,
            is_cosine_sim=True,
        )
        r_precision += calculate_R_precision(
            text_np,
            generated_np,
            top_k=3,
            sum_all=True,
            is_cosine_sim=True,
        )
        matching_real += cosine_similarity_matrix(text_np, ground_truth_np).trace()
        matching += cosine_similarity_matrix(text_np, generated_np).trace()
        sample_count += len(motions)

    ground_truth_np = torch.cat(ground_truth_embeddings).cpu().numpy()
    generated_np = torch.cat(generated_embeddings).cpu().numpy()
    ground_truth_mean, ground_truth_covariance = calculate_activation_statistics(
        ground_truth_np
    )
    generated_mean, generated_covariance = calculate_activation_statistics(generated_np)
    fid = calculate_frechet_distance(
        ground_truth_mean,
        ground_truth_covariance,
        generated_mean,
        generated_covariance,
    )

    diversity_times = min(300, sample_count - 1)
    diversity_real = calculate_diversity(ground_truth_np, diversity_times)
    diversity = calculate_diversity(generated_np, diversity_times)
    r_precision_real /= sample_count
    r_precision /= sample_count
    matching_real /= sample_count
    matching /= sample_count

    multimodality = 0.0
    if multimodality_embeddings:
        multimodality_np = torch.cat(multimodality_embeddings).cpu().numpy()
        multimodality = calculate_multimodality(multimodality_np, 10)

    print(
        f"--> \t SnapMoGen Eva. Ep/Re {ep}: FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {r_precision_real}, R_precision. {r_precision}, "
        f"matching_real. {matching_real:.4f}, matching_pred. {matching:.4f}, "
        f"multimodality. {multimodality:.4f}"
    )

    best_fid = min(best_fid, fid)
    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        best_div = diversity
    best_top1 = max(best_top1, r_precision[0])
    best_top2 = max(best_top2, r_precision[1])
    best_top3 = max(best_top3, r_precision[2])
    best_matching = max(best_matching, matching)
    clip_score_old = max(clip_score_old, matching)

    current_metrics = {
        "fid": float(fid),
        "diversity": float(diversity),
        "top1": float(r_precision[0]),
        "top2": float(r_precision[1]),
        "top3": float(r_precision[2]),
        "matching": float(matching),
        "multimodality": float(multimodality),
        "clip_score": float(matching),
    }
    result = (
        best_fid,
        best_div,
        best_top1,
        best_top2,
        best_top3,
        best_matching,
        multimodality if cal_mm else 0.0,
        clip_score_old,
    )
    if return_current:
        return result, current_metrics
    return result
