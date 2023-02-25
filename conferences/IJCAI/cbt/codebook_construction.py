"""
This code has been adapted from this repository:
https://github.com/Storyboardslee/codebook-transfer-learning

Implemented from Eq.(2) of
Li, Bin & Yang, Qiang & Xue, Xiangyang. (2009).
Can Movies and Books Collaborate? Cross-Domain Collaborative Filtering for Sparsity Reduction..
IJCAI International Joint Conference on Artificial Intelligence. 2052-2057.
"""
import numpy as np
from numpy.linalg import multi_dot


def codebook_construction(X, F, G, binarize):
    if binarize:
        F = binarize_matrix(F)
        G = binarize_matrix(G)

    sum_vector = np.ones(X.shape)
    enum = multi_dot([F.T, X, G])
    denom = multi_dot([F.T, sum_vector, G])

    B = np.nan_to_num(enum / denom)

    print('')
    print('CODEBOOK')
    print(B)

    return B


def binarize_matrix(X):
    """
    Binarize input matrix with 0 and 1
    """
    X_copy = np.copy(X)
    for i in range(X_copy.shape[0]):
        # Find largest score in the row
        max_score = sorted(list(X_copy[i, :])).pop()
        # Replace all other scores into 0
        X_copy[i, :] = np.where(X_copy[i, :] == max_score, X_copy[i, :], 0)
        # Replace largest score with 1
        X_copy[i, :] = np.where(X_copy[i, :] == 0, X_copy[i, :], 1)

    return X_copy
