import numpy as np
import torch
from tqdm import tqdm

from utils.back_process import back_process, back_process_full
from utils.eval_msflow_263 import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
    calculate_R_precision,
    euclidean_distance_matrix,
)
from utils.raw_joint_utils import as_numpy, unflatten_raw_joints_np


def _raw_to_eval_features(raw_motions, dataset, eval_mean, eval_std, dataset_name):
    feature_converter = back_process if eval_mean.shape[0] == 67 else back_process_full
    features = []
    for batch_index in range(raw_motions.shape[0]):
        features.append(
            feature_converter(raw_motions[batch_index], dataset_name=dataset_name)
        )
    features = np.stack(features, axis=0)
    return dataset.transform(features, eval_mean, eval_std)


def _normalized_raw_to_eval_features(
    motion,
    dataset,
    eval_mean,
    eval_std,
    dataset_name,
    raw_joint_count,
    raw_joint_dim,
):
    raw_motion = unflatten_raw_joints_np(
        as_numpy(motion),
        joint_count=raw_joint_count,
        joint_dim=raw_joint_dim,
    )
    raw_motion = dataset.inv_transform(raw_motion)
    return _raw_to_eval_features(raw_motion, dataset, eval_mean, eval_std, dataset_name)


@torch.no_grad()
def evaluation_msflow_xyz(
    val_loader,
    ema_model,
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
    cond_scale,
    cal_mm=False,
    eval_mean=None,
    eval_std=None,
    return_current=False,
    dataset_name=None,
    raw_joint_count=22,
    raw_joint_dim=3,
):
    ema_model.eval()
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    if cond_scale is None:
        cond_scale = 2.5
    clip_score_real = 0
    nb_sample = 0
    num_mm_batch = 3 if cal_mm else 0
    dataset_name = dataset_name or getattr(val_loader.dataset, "dataset_name", None)

    for batch_index, batch in enumerate(tqdm(val_loader)):
        if len(batch) == 8:
            (
                word_embeddings,
                pos_one_hots,
                clip_text,
                sent_len,
                pose,
                m_length,
                _,
                generation_text,
            ) = batch
        else:
            (
                word_embeddings,
                pos_one_hots,
                clip_text,
                sent_len,
                pose,
                m_length,
                _,
            ) = batch
            generation_text = clip_text
        m_length = m_length.to(device)
        bs = pose.shape[0]

        if batch_index < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30)):
                pred_motions = ema_model.generate(
                    generation_text,
                    m_length,
                    cond_scale,
                )
                pred_motions = _normalized_raw_to_eval_features(
                    pred_motions,
                    val_loader.dataset,
                    eval_mean,
                    eval_std,
                    dataset_name,
                    raw_joint_count,
                    raw_joint_dim,
                )
                (et_pred, em_pred), (
                    et_pred_clip,
                    em_pred_clip,
                ) = eval_wrapper.get_co_embeddings(
                    word_embeddings,
                    pos_one_hots,
                    sent_len,
                    clip_text,
                    torch.from_numpy(pred_motions).to(device),
                    m_length - 1,
                )
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)
            motion_multimodality.append(motion_multimodality_batch)
            for sample_index in range(bs):
                single_em = em_pred_clip[sample_index]
                single_et = et_pred_clip[sample_index]
                batch_clip_score_pred += (single_em @ single_et.T).item()
            clip_score_real += batch_clip_score_pred
        elif num_mm_batch == 0:
            pred_motions = ema_model.generate(
                generation_text,
                m_length,
                cond_scale,
            )
            pred_motions = _normalized_raw_to_eval_features(
                pred_motions,
                val_loader.dataset,
                eval_mean,
                eval_std,
                dataset_name,
                raw_joint_count,
                raw_joint_dim,
            )
            (et_pred, em_pred), (
                et_pred_clip,
                em_pred_clip,
            ) = eval_wrapper.get_co_embeddings(
                word_embeddings,
                pos_one_hots,
                sent_len,
                clip_text,
                torch.from_numpy(pred_motions).to(device),
                m_length - 1,
            )
            batch_clip_score_pred = 0
            for sample_index in range(bs):
                single_em = em_pred_clip[sample_index]
                single_et = et_pred_clip[sample_index]
                batch_clip_score_pred += (single_em @ single_et.T).item()
            clip_score_real += batch_clip_score_pred
        else:
            print("No multimodality batch")

        pose = _normalized_raw_to_eval_features(
            pose,
            val_loader.dataset,
            eval_mean,
            eval_std,
            dataset_name,
            raw_joint_count,
            raw_joint_dim,
        )
        pose = torch.from_numpy(pose).to(device).float()
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(
            word_embeddings,
            pos_one_hots,
            sent_len,
            clip_text,
            pose.clone(),
            m_length - 1,
        )
        batch_clip_score = 0
        for sample_index in range(bs):
            single_em = em_clip[sample_index]
            single_et = et_clip[sample_index]
            batch_clip_score += (single_em @ single_et.T).item()
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et.cpu().numpy(), em.cpu().numpy()
        ).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        temp_match = euclidean_distance_matrix(
            et_pred.cpu().numpy(), em_pred.cpu().numpy()
        ).trace()
        R_precision += temp_R
        matching_score_pred += temp_match
        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    clip_score_real = clip_score_real / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    print(
        f"--> \t Eva. Ep/Re {ep} :, FID. {fid:.4f}, Diversity Real. "
        f"{diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real}, "
        f"matching_score_pred. {matching_score_pred} "
        f"multimodality. {multimodality:.4f}, clip score. {clip_score_real}"
    )

    if fid < best_fid:
        best_fid = fid
    if matching_score_pred < best_matching:
        best_matching = matching_score_pred
    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        best_div = diversity
    if R_precision[0] > best_top1:
        best_top1 = R_precision[0]
    if R_precision[1] > best_top2:
        best_top2 = R_precision[1]
    if R_precision[2] > best_top3:
        best_top3 = R_precision[2]
    if clip_score_real > clip_score_old:
        clip_score_old = clip_score_real

    current_metrics = {
        "fid": float(fid),
        "diversity": float(diversity),
        "top1": float(R_precision[0]),
        "top2": float(R_precision[1]),
        "top3": float(R_precision[2]),
        "matching": float(matching_score_pred),
        "multimodality": float(multimodality),
        "clip_score": float(clip_score_real),
    }
    result = (
        best_fid,
        best_div,
        best_top1,
        best_top2,
        best_top3,
        best_matching,
        multimodality if cal_mm else 0,
        clip_score_old,
    )
    if return_current:
        return result, current_metrics
    return result
