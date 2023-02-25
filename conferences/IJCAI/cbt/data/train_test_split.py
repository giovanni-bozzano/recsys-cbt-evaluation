"""
This code has been adapted from this repository:
https://github.com/transferhp/cbt
"""

import numpy as np


def train_test_split(ratings, k, given_k):
    """
    Split input data into training and test data.
    The first k row users and their ratings are selected as training data,
    the remaining users and their ratings are left to test data.
    For each test user, randomly select given_k observed ratings and provide them to training set,
    and the remaining ratings are used for evaluation.


    Parameters
    ----------------
    ratings : sparse csr_matrix
    The input n * m rating matrix.

    k : int
    Top k users are selected for training set,
    then n - k users are left for test.

    given_k : int
    For each test user, give_k observed ratings are provided to training.

    Returns
    -----------------
    train : sparse csr_matrix
    A n * m rating matrix for training data.

    test : sparse csr_matrix
    A n * m rating matrix for test data.

    """
    train = ratings.copy()
    test = ratings.copy()

    # change sparsity structure
    train = train.tolil()
    train[k:, :] = 0.  # only keep first k rows
    test = test.tolil()
    test[:k, :] = 0.  # only keep remaining rows

    for row in range(k, test.shape[0]):
        col_indices = np.random.choice(ratings[row, :].nonzero()[1], size=given_k, replace=False)
        for col in col_indices:
            train[row, col] = test[row, col]
            test[row, col] = 0.

    train = train.tocsr()
    test = test.tocsr()

    # Test and training are truly disjoint
    assert train.multiply(test).nnz == 0  # sparse format
    return train, test
