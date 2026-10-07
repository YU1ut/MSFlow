"""SnapMoGen retrieval metrics; distribution metrics use the HumanML3D helpers."""

import numpy as np

from utils.eval_msflow_263 import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_multimodality,
    calculate_top_k,
    euclidean_distance_matrix,
)


def cosine_similarity_matrix(matrix1, matrix2):
    matrix1 = matrix1 / np.linalg.norm(matrix1, axis=-1, keepdims=True)
    matrix2 = matrix2 / np.linalg.norm(matrix2, axis=-1, keepdims=True)
    return np.dot(matrix1, matrix2.T)


def calculate_R_precision(
    embedding1, embedding2, top_k, sum_all=False, is_cosine_sim=False
):
    if is_cosine_sim:
        distances = -cosine_similarity_matrix(embedding1, embedding2)
    else:
        distances = euclidean_distance_matrix(embedding1, embedding2)
    top_k_matrix = calculate_top_k(np.argsort(distances, axis=1), top_k)
    return top_k_matrix.sum(axis=0) if sum_all else top_k_matrix
