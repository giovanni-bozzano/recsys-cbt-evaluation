from conferences.IJCAI.cbt.datasets_provided.ml_sparse_to_bx_reader import MovieLensSparseToBookCrossingReader
from Data_manager.DataReader import DataReader
from Data_manager.Dataset import Dataset


class MovieLensSparseToBookCrossingReaderWrapper(DataReader):
    DATASET_SUBFOLDER = 'MovieLensSparseToBookCrossingReader_Wrapper/'
    AVAILABLE_URM = ['URM_all']
    IS_IMPLICIT = False

    def __init__(self):
        super(MovieLensSparseToBookCrossingReaderWrapper, self).__init__()
        self._originalReader = MovieLensSparseToBookCrossingReader()
        self.URM_source = self._originalReader.URM_dictionary['URM_source']

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        URM_all = self._originalReader.URM_dictionary['URM_target_train'] + \
                  self._originalReader.URM_dictionary['URM_target_validation'] + \
                  self._originalReader.URM_dictionary['URM_target_test']

        loaded_URM_dict = {'URM_all': URM_all}

        print('================')
        print('SOURCE MovieLens 1M Sparse')
        n_users, n_items = self.URM_source.shape
        print('Reduced URM shape: {0} * {1}'.format(n_users, n_items))
        print('Reduced URM ratings: {0}'.format(float(self.URM_source.nnz)))
        sparsity = float(self.URM_source.nnz)
        sparsity /= (n_users * n_items)
        print('Reduced URM density: {:.4f}%'.format(sparsity * 100.))

        print('TARGET BookCrossing')
        n_users, n_items = URM_all.shape
        print('Reduced URM shape: {0} * {1}'.format(n_users, n_items))
        print('Reduced URM ratings: {0}'.format(float(URM_all.nnz)))
        sparsity = float(URM_all.nnz)
        sparsity /= (n_users * n_items)
        print('Reduced URM density: {:.4f}%'.format(sparsity * 100.))
        print('================')

        user_original_ID_to_index = {i: i for i in range(n_users)}
        item_original_ID_to_index = {i: i for i in range(n_items)}

        loaded_dataset = Dataset(dataset_name=self._get_dataset_name(),
                                 URM_dictionary=loaded_URM_dict,
                                 ICM_dictionary=None,
                                 ICM_feature_mapper_dictionary=None,
                                 UCM_dictionary=None,
                                 UCM_feature_mapper_dictionary=None,
                                 user_original_ID_to_index=user_original_ID_to_index,
                                 item_original_ID_to_index=item_original_ID_to_index,
                                 is_implicit=self.IS_IMPLICIT)

        return loaded_dataset
