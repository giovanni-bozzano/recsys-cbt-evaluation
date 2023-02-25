import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'framework'))

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

from conferences.IJCAI.cbt.cbt import CBTRecommender
from conferences.IJCAI.cbt.data.parse_data import fill_row_miss_value
from conferences.IJCAI.cbt.datasets_provided.movielens_hetrec_2011_reduced_reader import MovielensHetrec2011ReducedReader
from framework.Data_manager.DataSplitter_k_fold_random import DataSplitter_k_fold_random_fromDataSplitter
from framework.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from framework.Data_manager.Movielens.MovielensHetrec2011Reader import MovielensHetrec2011Reader
from framework.Data_manager.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader
from framework.Data_manager.Utility import filter_urm


dataSplitter_kwargs = {
    'k_out_value': 1,
    'use_validation_set': True,
    'leave_random_out': True
}

output_folder_path = 'result_experiments/plots/movielens/'
metric_to_optimize = 'MAP'
k_out_value = 1
n_cases = 1
n_random_starts = 1
cutoff_list_test = [20]

source_data_reader = NetflixPrizeReader()
source_loaded_dataset = source_data_reader.load_data()
URM_source = source_loaded_dataset.AVAILABLE_URM['URM_all'].copy()
for i in range(10):
    URM_source = filter_urm(URM_source, user_min_number_ratings=1075, item_min_number_ratings=1075)
URM_source = URM_source[:500, :500]
URM_source = URM_source.toarray()

target_data_reader = MovielensHetrec2011ReducedReader(dense=True, min_user_interactions=k_out_value * 2 + 1)

dataSplitter_k_fold = DataSplitter_k_fold_random_fromDataSplitter(target_data_reader, DataSplitter_leave_k_out,
                                                                  dataSplitter_kwargs=dataSplitter_kwargs,
                                                                  n_folds=2,
                                                                  preload_all=False)

dataSplitter_k_fold.load_data(save_folder_path=output_folder_path + 'data/plots/')

for fold_index, dataSplitter_fold in enumerate(dataSplitter_k_fold):

    URM_source_processed = URM_source.copy()

    URM_source_processed = sps.csr_matrix(URM_source_processed).astype(np.float32)
    URM_source_processed = fill_row_miss_value(URM_source_processed)
    URM_source_processed = sps.csr_matrix(URM_source_processed).astype(np.float32)
    URM_source_processed.eliminate_zeros()

    URM_target_train, _, _ = dataSplitter_fold.get_holdout_split()
    URM_target_train.eliminate_zeros()
    URM_target_train = URM_target_train.astype(np.float32)

    plt.figure(figsize=(15, 8))
    plt.xlabel('Items')
    plt.ylabel('Users')
    plt.imshow(URM_target_train.toarray(), cmap='hot', interpolation='nearest')
    plt.savefig('plots/movielens-target.png')
    plt.close()

    cbt = CBTRecommender(URM_target_train=URM_target_train, URM_source=URM_source_processed)
    cbt.fit(50, 50, 1000, 1, 5)

    print('======================================')
    print(cbt.B_codebook)
    print(cbt.URM_target_train_filled.toarray().size - np.count_nonzero(cbt.URM_target_train_filled.toarray()))
    plt.figure(figsize=(15, 8))
    plt.xlabel('Items')
    plt.ylabel('Users')
    plt.imshow(cbt.URM_target_train_filled.toarray(), cmap='hot', interpolation='nearest')
    plt.savefig('plots/movielens-target-filled.png')
    plt.close()


output_folder_path = 'result_experiments/plots/movielens-full/'
metric_to_optimize = 'MAP'
k_out_value = 1
n_cases = 1
n_random_starts = 1
cutoff_list_test = [20]

source_data_reader = NetflixPrizeReader()
source_loaded_dataset = source_data_reader.load_data()
URM_source = source_loaded_dataset.AVAILABLE_URM['URM_all'].copy()
for i in range(10):
    URM_source = filter_urm(URM_source, user_min_number_ratings=1075, item_min_number_ratings=1075)
URM_source = URM_source[:500, :500]
URM_source = URM_source.toarray()

target_data_reader = MovielensHetrec2011Reader()

dataSplitter_k_fold = DataSplitter_k_fold_random_fromDataSplitter(target_data_reader, DataSplitter_leave_k_out,
                                                                  dataSplitter_kwargs=dataSplitter_kwargs,
                                                                  n_folds=2,
                                                                  preload_all=False)

dataSplitter_k_fold.load_data(save_folder_path=output_folder_path + 'data/plots/')

for fold_index, dataSplitter_fold in enumerate(dataSplitter_k_fold):

    URM_source_processed = URM_source.copy()

    URM_source_processed = sps.csr_matrix(URM_source_processed).astype(np.float32)
    URM_source_processed = fill_row_miss_value(URM_source_processed)
    URM_source_processed = sps.csr_matrix(URM_source_processed).astype(np.float32)
    URM_source_processed.eliminate_zeros()

    URM_target_train, _, _ = dataSplitter_fold.get_holdout_split()
    URM_target_train.eliminate_zeros()
    URM_target_train = URM_target_train.astype(np.float32)

    plt.figure(figsize=(15, 8))
    plt.xlabel('Items')
    plt.ylabel('Users')
    plt.imshow(URM_target_train.toarray(), cmap='hot', interpolation='nearest')
    plt.savefig('plots/movielens-full-target.png')
    plt.close()

    cbt = CBTRecommender(URM_target_train=URM_target_train, URM_source=URM_source_processed)
    cbt.fit(50, 50, 1000, 1, 5)

    plt.figure(figsize=(15, 8))
    plt.xlabel('Items')
    plt.ylabel('Users')
    plt.imshow(cbt.URM_target_train_filled.toarray(), cmap='hot', interpolation='nearest')
    plt.savefig('plots/movielens-full-target-filled.png')
    plt.close()
