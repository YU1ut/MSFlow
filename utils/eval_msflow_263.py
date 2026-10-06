import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm


#################################################################################
#                               Eval Function Loops                             #
#################################################################################
@torch.no_grad()
def evaluation_msflow_263(
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
    clip_score_real = 0
    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(tqdm(val_loader)):
        (
            word_embeddings,
            pos_one_hots,
            clip_text,
            sent_len,
            pose,
            m_length,
            _,
        ) = batch
        m_length = m_length.to(device)

        bs = pose.shape[0]
        if i < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30)):
                pred_motions = ema_model.generate(
                    clip_text,
                    m_length,
                    cond_scale,
                )

                pred_motions = val_loader.dataset.inv_transform(
                    pred_motions.detach().cpu().numpy(),
                    mean=val_loader.dataset.mean[: pred_motions.shape[-1]],
                    std=val_loader.dataset.std[: pred_motions.shape[-1]],
                )
                if pred_motions.ndim == 3:
                    pred_motions = pred_motions[:, :, None, :]
                pred_motions_list = []
                for j in range(bs):
                    feature_dim = eval_mean.shape[0]
                    pred_motions_list.append(pred_motions[j, :, 0, :feature_dim])

                pred_motions = np.stack(pred_motions_list, axis=0)
                pred_motions = val_loader.dataset.transform(
                    pred_motions, eval_mean, eval_std
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
            motion_multimodality_batch = torch.cat(
                motion_multimodality_batch, dim=1
            )  # (bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        else:
            pred_motions = ema_model.generate(
                clip_text,
                m_length,
                cond_scale,
            )

            pred_motions = val_loader.dataset.inv_transform(
                pred_motions.detach().cpu().numpy(),
                mean=val_loader.dataset.mean[: pred_motions.shape[-1]],
                std=val_loader.dataset.std[: pred_motions.shape[-1]],
            )
            if pred_motions.ndim == 3:
                pred_motions = pred_motions[:, :, None, :]

            pred_motions_list = []
            for j in range(bs):
                feature_dim = eval_mean.shape[0]
                pred_motions_list.append(pred_motions[j, :, 0, :feature_dim])
            pred_motions = np.stack(pred_motions_list, axis=0)
            pred_motions = val_loader.dataset.transform(
                pred_motions, eval_mean, eval_std
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
            for j in range(bs):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred

        pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
        if pose.ndim == 3:
            pose = pose[:, :, None, :]
        poses = []
        for j in range(bs):
            feature_dim = eval_mean.shape[0]
            poses.append(pose[j, :, 0, :feature_dim])
        poses = np.stack(poses, axis=0)
        pose = val_loader.dataset.transform(poses, eval_mean, eval_std)
        pose = torch.from_numpy(pose).to(device).float()
        (et, em), _ = eval_wrapper.get_co_embeddings(
            word_embeddings,
            pos_one_hots,
            sent_len,
            clip_text,
            pose.clone(),
            m_length - 1,
        )
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

    msg = (
        f"--> \t Eva. Ep/Re {ep} :, FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real}, "
        f"matching_score_pred. {matching_score_pred} "
        f"multimodality. {multimodality:.4f}, clip score. {clip_score_real}"
    )
    print(msg)

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

    if cal_mm:
        result = (
            best_fid,
            best_div,
            best_top1,
            best_top2,
            best_top3,
            best_matching,
            multimodality,
            clip_score_old,
        )
    else:
        result = (
            best_fid,
            best_div,
            best_top1,
            best_top2,
            best_top3,
            best_matching,
            0,
            clip_score_old,
        )

    if return_current:
        return result, current_metrics
    return result


def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = correct_vec | bool_mat[:, i]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
