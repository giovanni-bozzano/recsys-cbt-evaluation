import shutil
import zipfile

from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, load_CSV_into_SparseBuilder
from Data_manager.Dataset import Dataset
from Data_manager.Utility import filter_urm


class MovielensHetrec2011ReducedReader(DataReader):
    DATASET_URL = 'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip'
    DATASET_SUBFOLDER = 'MovielensHetrec2011Reduced/'
    AVAILABLE_ICM = []

    IS_IMPLICIT = False

    def __init__(self, dense, min_user_interactions):
        super(MovielensHetrec2011ReducedReader, self).__init__()
        self.dense = dense
        self.min_user_interactions = min_user_interactions
        if dense:
            self.DATASET_SUBFOLDER = 'MovielensHetrec2011Reduced/dense/'
        else:
            self.DATASET_SUBFOLDER = 'MovielensHetrec2011Reduced/sparse/'

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original

        zipFile_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:
            dataFile = zipfile.ZipFile(zipFile_path + 'hetrec2011-movielens-2k-v2.zip')

        except (FileNotFoundError, zipfile.BadZipFile):
            self._print('Unable to fild data zip file. Downloading...')
            download_from_URL(self.DATASET_URL, zipFile_path, 'hetrec2011-movielens-2k-v2.zip')
            dataFile = zipfile.ZipFile(zipFile_path + 'hetrec2011-movielens-2k-v2.zip')

        URM_path = dataFile.extract('user_ratedmovies.dat', path=zipFile_path + 'decompressed/')

        URM_all, item_original_ID_to_index, user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator='\t', header=True,
                                                                                                    custom_user_item_rating_columns=[0, 1, 2])

        n_users, n_items = URM_all.shape
        print('MovieLens Hetrec 2011, Dense: ' + str(self.dense))
        print('Original URM shape: {0} * {1}'.format(n_users, n_items))
        print('Original URM ratings: {0}'.format(float(URM_all.nnz)))
        sparsity = float(URM_all.nnz)
        sparsity /= (n_users * n_items)
        print('Original URM density: {:.4f}%'.format(sparsity * 100.))

        if self.dense:
            # Density: 11.6596%
            for i in range(5):
                URM_all = filter_urm(URM_all, user_min_number_ratings=50, item_min_number_ratings=50)
            URM_all = URM_all[:, 1000:2000]
            URM_all = filter_urm(URM_all, user_min_number_ratings=self.min_user_interactions, item_min_number_ratings=0)
            URM_all = URM_all[500:1000, :]
        else:
            # Density: 2.9254%
            for i in range(5):
                URM_all = filter_urm(URM_all, user_min_number_ratings=25, item_min_number_ratings=25)
            URM_all = URM_all[:, -1000:]
            URM_all = filter_urm(URM_all, user_min_number_ratings=self.min_user_interactions, item_min_number_ratings=0)
            URM_all = URM_all[-500:, :]
        n_users, n_items = URM_all.shape
        print('Reduced URM shape: {0} * {1}'.format(n_users, n_items))
        print('Reduced URM ratings: {0}'.format(float(URM_all.nnz)))
        sparsity = float(URM_all.nnz)
        sparsity /= (n_users * n_items)
        print('Reduced URM density: {:.4f}%'.format(sparsity * 100.))
        user_original_ID_to_index = {i: i for i in range(n_users)}
        item_original_ID_to_index = {i: i for i in range(n_items)}

        loaded_URM_dict = {'URM_all': URM_all}

        loaded_dataset = Dataset(dataset_name=self._get_dataset_name(),
                                 URM_dictionary=loaded_URM_dict,
                                 user_original_ID_to_index=user_original_ID_to_index,
                                 item_original_ID_to_index=item_original_ID_to_index,
                                 is_implicit=self.IS_IMPLICIT)

        self._print('cleaning temporary files')

        shutil.rmtree(zipFile_path + 'decompressed', ignore_errors=True)

        self._print('loading complete')

        return loaded_dataset
