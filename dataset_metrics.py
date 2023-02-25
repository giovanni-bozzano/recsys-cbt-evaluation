import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'framework'))

from conferences.IJCAI.cbt.datasets_provided.amazon_movies_tv_reduced_reader import AmazonMoviesTVReducedReader
from conferences.IJCAI.cbt.datasets_provided.ml_dense_to_bx_reader_wrapper import MovieLensDenseToBookCrossingReaderWrapper
from conferences.IJCAI.cbt.datasets_provided.ml_dense_to_nf_dense_reader_wrapper import MovieLensDenseToNetflixDenseReaderWrapper
from conferences.IJCAI.cbt.datasets_provided.ml_sparse_to_bx_reader_wrapper import MovieLensSparseToBookCrossingReaderWrapper
from conferences.IJCAI.cbt.datasets_provided.ml_sparse_to_nf_sparse_reader_wrapper import MovieLensSparseToNetflixSparseReaderWrapper
from conferences.IJCAI.cbt.datasets_provided.movielens_hetrec_2011_reduced_reader import MovielensHetrec2011ReducedReader
from conferences.IJCAI.cbt.datasets_provided.netflix_prize_reduced_reader import NetflixPrizeReducedReader
from conferences.IJCAI.cbt.datasets_provided.nf_dense_to_bx_reader_wrapper import NetflixDenseToBookCrossingReaderWrapper
from conferences.IJCAI.cbt.datasets_provided.nf_dense_to_ml_dense_reader_wrapper import NetflixDenseToMovieLensDenseReaderWrapper
from conferences.IJCAI.cbt.datasets_provided.nf_sparse_to_bx_reader_wrapper import NetflixSparseToBookCrossingReaderWrapper
from conferences.IJCAI.cbt.datasets_provided.nf_sparse_to_ml_sparse_reader_wrapper import NetflixSparseToMovieLensSparseReaderWrapper
from framework.Data_manager.Movielens.Movielens20MReader import Movielens20MReader
from framework.Data_manager.Movielens.MovielensHetrec2011Reader import MovielensHetrec2011Reader
from framework.Data_manager.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader
from framework.Data_manager.Utility import filter_urm


print('========================================================')
print('OLD DATASETS')
print('========================================================\n\n\n\n\n\n')

data_reader = MovieLensDenseToBookCrossingReaderWrapper()
data_reader._load_from_original_file()
data_reader = MovieLensSparseToBookCrossingReaderWrapper()
data_reader._load_from_original_file()
data_reader = MovieLensDenseToNetflixDenseReaderWrapper()
data_reader._load_from_original_file()
data_reader = MovieLensSparseToNetflixSparseReaderWrapper()
data_reader._load_from_original_file()
data_reader = NetflixDenseToBookCrossingReaderWrapper()
data_reader._load_from_original_file()
data_reader = NetflixSparseToBookCrossingReaderWrapper()
data_reader._load_from_original_file()
data_reader = NetflixDenseToMovieLensDenseReaderWrapper()
data_reader._load_from_original_file()
data_reader = NetflixSparseToMovieLensSparseReaderWrapper()
data_reader._load_from_original_file()

print('\n\n\n\n\n\n========================================================')
print('NEW DATASETS')
print('========================================================\n\n\n\n\n\n')

target_data_reader = MovielensHetrec2011Reader()
target_loaded_dataset = target_data_reader.load_data()
URM_target = target_loaded_dataset.AVAILABLE_URM['URM_all'].copy()
n_users, n_items = URM_target.shape
print('TARGET MovieLens Hetrec 2011')
print('Original URM shape: {0} * {1}'.format(n_users, n_items))
print('Original URM ratings: {0}'.format(float(URM_target.nnz)))
sparsity = float(URM_target.nnz)
sparsity /= (n_users * n_items)
print('Original URM density: {:.4f}%'.format(sparsity * 100.))

source_data_reader = Movielens20MReader()
source_loaded_dataset = source_data_reader.load_data()
URM_source = source_loaded_dataset.AVAILABLE_URM['URM_all'].copy()
n_users, n_items = URM_source.shape
print('SOURCE MovieLens 20M')
print('Original URM shape: {0} * {1}'.format(n_users, n_items))
print('Original URM ratings: {0}'.format(float(URM_source.nnz)))
sparsity = float(URM_source.nnz)
sparsity /= (n_users * n_items)
print('Original URM density: {:.4f}%'.format(sparsity * 100.))
for i in range(4):
    URM_source = filter_urm(URM_source, user_min_number_ratings=645, item_min_number_ratings=645)
URM_source = URM_source[:500, :500]
n_users, n_items = URM_source.shape
print('Reduced URM shape: {0} * {1}'.format(n_users, n_items))
print('Reduced URM ratings: {0}'.format(float(URM_source.nnz)))
sparsity = float(URM_source.nnz)
sparsity /= (n_users * n_items)
print('Reduced URM density: {:.4f}%'.format(sparsity * 100.))

source_data_reader = NetflixPrizeReader()
source_loaded_dataset = source_data_reader.load_data()
URM_source = source_loaded_dataset.AVAILABLE_URM['URM_all'].copy()
n_users, n_items = URM_source.shape
print('SOURCE Netflix Prize')
print('Original URM shape: {0} * {1}'.format(n_users, n_items))
print('Original URM ratings: {0}'.format(float(URM_source.nnz)))
sparsity = float(URM_source.nnz)
sparsity /= (n_users * n_items)
print('Original URM density: {:.4f}%'.format(sparsity * 100.))
for i in range(10):
    URM_source = filter_urm(URM_source, user_min_number_ratings=1075, item_min_number_ratings=1075)
URM_source = URM_source[:500, :500]
n_users, n_items = URM_source.shape
print('Reduced URM shape: {0} * {1}'.format(n_users, n_items))
print('Reduced URM ratings: {0}'.format(float(URM_source.nnz)))
sparsity = float(URM_source.nnz)
sparsity /= (n_users * n_items)
print('Reduced URM density: {:.4f}%'.format(sparsity * 100.))

target_data_reader = MovielensHetrec2011ReducedReader(dense=True, min_user_interactions=1 * 2 + 1)
target_data_reader._load_from_original_file()

target_data_reader = MovielensHetrec2011ReducedReader(dense=False, min_user_interactions=1 * 2 + 1)
target_data_reader._load_from_original_file()

target_data_reader = NetflixPrizeReducedReader(dense=True, min_user_interactions=1 * 2 + 1)
target_data_reader._load_from_original_file()

target_data_reader = NetflixPrizeReducedReader(dense=False, min_user_interactions=1 * 2 + 1)
target_data_reader._load_from_original_file()

target_data_reader = AmazonMoviesTVReducedReader(dense=True, min_user_interactions=1 * 2 + 1)
target_data_reader._load_from_original_file()

target_data_reader = AmazonMoviesTVReducedReader(dense=False, min_user_interactions=1 * 2 + 1)
target_data_reader._load_from_original_file()
