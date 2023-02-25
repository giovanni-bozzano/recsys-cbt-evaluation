"""
This code has been adapted from this repository:
https://github.com/transferhp/cbt
"""

import random

import numpy as np
import scipy.io as sio
import scipy.sparse as sps


def fill_row_miss_value(arr_csr):
    """
    For each row in arr_csr,
    fill the missing value with mean value of non-zero elements in this row.

    Parameters
    ----------------
    arr_csr : sparse.csr_matrix
    The input sparse matrix.

    Returns
    ----------------
    Dense 2*D array after filling values to missing positions.
    """
    arr = arr_csr.toarray()
    all_cols = np.arange(arr_csr.shape[1])
    for i in range(len(arr_csr.indptr) - 1):
        row_sum = arr_csr[i, :].sum() / float(arr_csr[i, :].nnz)
        start = arr_csr.indptr[i]
        end = arr_csr.indptr[i + 1]
        arr[i, np.setdiff1d(all_cols, arr_csr.indices[start: end])] = row_sum
    return arr


def parse(source_path, source_name, target_path, target_name):
    # Load source data: EachMovie
    URM_source_original = sio.loadmat(source_path)[source_name].astype(np.float32)
    URM_source_original = sps.csr_matrix(URM_source_original)
    # Randomly select 500 users and items
    users_partition = random.sample(range(URM_source_original.shape[0]), 500)
    items_partition = random.sample(range(URM_source_original.shape[1]), 500)
    # Slicing source matrix
    URM_source_original = URM_source_original[users_partition, :].tocsc()[:, items_partition].tocsr()
    URM_source_original.eliminate_zeros()
    # Count statistics of source dataset
    n_users = URM_source_original.shape[0]
    n_items = URM_source_original.shape[1]
    print('Source rating matrix shape: {0} * {1}'.format(n_users, n_items))
    sparsity = float(URM_source_original.nnz)
    sparsity /= (n_users * n_items)
    print('Density in source dataset: {:.4f}%'.format(sparsity * 100.))
    print(20 * '--')
    # Fill missing values
    # URM_source = fill_row_miss_value(URM_source_original)
    URM_source = URM_source_original
    URM_source = sps.csr_matrix(URM_source)
    URM_source.eliminate_zeros()

    # Load target data
    URM_target = sio.loadmat(target_path)[target_name].astype(np.float32)
    URM_target = sps.csr_matrix(URM_target)
    # Count statistics of target dataset
    n_users = URM_target.shape[0]
    n_items = URM_target.shape[1]
    print('Target rating matrix shape: {0} * {1}'.format(n_users, n_items))
    sparsity = float(URM_target.nnz)
    sparsity /= (n_users * n_items)
    print('Density in target dataset: {:.4f}%'.format(sparsity * 100.))
    print(20 * '--')

    return URM_source, URM_target
