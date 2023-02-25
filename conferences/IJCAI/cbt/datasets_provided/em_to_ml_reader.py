import numpy as np

from conferences.IJCAI.cbt.data import parse_data_original
from conferences.IJCAI.cbt.data.train_test_split import train_test_split
from Data_manager.split_functions.split_train_validation import split_train_validation_percentage_user_wise


class EachMovieToMovieLensReader(object):
    URM_dictionary = {}

    def __init__(self):
        super(EachMovieToMovieLensReader, self).__init__()

        original_data_path = 'conferences/IJCAI/cbt/data/'

        URM_source, URM_target = parse_data_original.parse(original_data_path + 'eachmovie.mat', 'eachmovie',
                                                           original_data_path + 'ml100k.data')
        URM_target_train, URM_target_test = train_test_split(URM_target, k=300, given_k=15)
        URM_target_train, URM_target_validation = split_train_validation_percentage_user_wise(URM_target_train, train_percentage=0.8)
        URM_target_train = URM_target_train.astype(np.float32)
        URM_target_validation = URM_target_validation.astype(np.float32)

        self.URM_dictionary = {
            'URM_source': URM_source,
            'URM_target_train': URM_target_train,
            'URM_target_validation': URM_target_validation,
            'URM_target_test': URM_target_test
        }

        print('EachMovieToMovieLensReader: loading complete')
