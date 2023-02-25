from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import load_CSV_into_SparseBuilder, download_from_URL
from Data_manager.Dataset import Dataset
from Data_manager.Utility import filter_urm


class AmazonMoviesTVReducedReader(DataReader):
    DATASET_URL_RATING = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv'
    DATASET_SUBFOLDER = 'AmazonReviewData/AmazonMoviesTVReduced/'
    AVAILABLE_ICM = []

    IS_IMPLICIT = False

    def __init__(self, dense, min_user_interactions):
        super(AmazonMoviesTVReducedReader, self).__init__()
        self.dense = dense
        self.min_user_interactions = min_user_interactions
        if dense:
            self.DATASET_SUBFOLDER = 'AmazonReviewData/AmazonMoviesTVReduced/dense/'
        else:
            self.DATASET_SUBFOLDER = 'AmazonReviewData/AmazonMoviesTVReduced/sparse/'

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original

        dataset_split_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:
            open(dataset_split_folder + 'ratings_Movies_and_TV.csv', 'r')
        except FileNotFoundError:
            self._print('Unable to find or open review file. Downloading...')
            download_from_URL(self.DATASET_URL_RATING, dataset_split_folder, 'ratings_Movies_and_TV.csv')
        URM_path = dataset_split_folder + 'ratings_Movies_and_TV.csv'

        URM_all, _, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator=',', header=False, timestamp=True)

        n_users, n_items = URM_all.shape
        print('Amazon Movies and TV Series, Dense: ' + str(self.dense))
        print('Original URM shape: {0} * {1}'.format(n_users, n_items))
        print('Original URM ratings: {0}'.format(float(URM_all.nnz)))
        sparsity = float(URM_all.nnz)
        sparsity /= (n_users * n_items)
        print('Original URM density: {:.4f}%'.format(sparsity * 100.))

        if self.dense:
            # Density: 10.2096%
            for i in range(5):
                URM_all = filter_urm(URM_all, user_min_number_ratings=45, item_min_number_ratings=45)
            URM_all = URM_all[:, :1000]
            URM_all = filter_urm(URM_all, user_min_number_ratings=self.min_user_interactions, item_min_number_ratings=0)
            URM_all = URM_all[:500, :]
        else:
            # Density: 2.8832%
            for i in range(5):
                URM_all = filter_urm(URM_all, user_min_number_ratings=25, item_min_number_ratings=25)
            URM_all = URM_all[:, :1000]
            URM_all = filter_urm(URM_all, user_min_number_ratings=self.min_user_interactions, item_min_number_ratings=0)
            URM_all = URM_all[:500, :]
        n_users, n_items = URM_all.shape
        print('Reduced URM shape: {0} * {1}'.format(n_users, n_items))
        print('Reduced URM ratings: {0}'.format(float(URM_all.nnz)))
        sparsity = float(URM_all.nnz)
        sparsity /= (n_users * n_items)
        print('Reduced URM density: {:.4f}%'.format(sparsity * 100.))
        self.user_original_ID_to_index = {i: i for i in range(n_users)}
        self.item_original_ID_to_index = {i: i for i in range(n_items)}

        loaded_URM_dict = {'URM_all': URM_all}

        loaded_dataset = Dataset(dataset_name=self._get_dataset_name(),
                                 URM_dictionary=loaded_URM_dict,
                                 user_original_ID_to_index=self.user_original_ID_to_index,
                                 item_original_ID_to_index=self.item_original_ID_to_index,
                                 is_implicit=self.IS_IMPLICIT)

        self._print('loading complete')

        return loaded_dataset
