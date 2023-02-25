import time

import numpy as np
import scipy.sparse as sps

from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from Base.DataIO import DataIO
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.Recommender_utils import check_matrix
from Base.Similarity.Compute_Similarity import Compute_Similarity
from conferences.IJCAI.cbt.codebook_construction import codebook_construction
from conferences.IJCAI.cbt.codebook_transfer import CodebookTransfer
from conferences.IJCAI.cbt.onmtf import ONMTF


class CBTRecommender(BaseSimilarityMatrixRecommender, Incremental_Training_Early_Stopping):
    _iteration_loss = None

    RECOMMENDER_NAME = 'CBTRecommender'

    def __init__(self, URM_target_train, URM_source, baseline=False, verbose=True):
        super(CBTRecommender, self).__init__(URM_target_train, verbose)

        if baseline:
            self.RECOMMENDER_NAME = 'CBTBaselineRecommender'

        self.URM_source = check_matrix(URM_source.copy(), 'csr', dtype=np.float32).toarray()

        self.onmtf = None
        self.F_best = None
        self.G_best = None
        self.construction_loss = None
        self.transfer_loss = None
        self.URM_target_train_filled = None
        self.W_sparse = None

    def fit(self,
            n_user_clusters=50,
            n_item_clusters=50,
            epochs=20000,  # maximum_construct_iterations
            transfer_attempts=3,
            maximum_fill_iterations=100,

            # early-stopping parameters
            es_validation_every_n=1,
            es_stop_on_validation=True,
            es_lower_validations_allowed=2000,
            es_evaluator_object=None,
            es_validation_metric='loss',

            topK=20,
            shrink=0,
            similarity='pearson',
            normalize=False,
            feature_weighting='none',
            **similarity_parameters):
        assert n_user_clusters <= self.URM_source.shape[0], 'n_user_clusters is bigger than the number of source users'
        assert n_item_clusters <= self.URM_source.shape[1], 'n_item_clusters is bigger than the number of source items'

        # Codebook construction
        self.onmtf = ONMTF(self.URM_source, n_user_clusters, n_item_clusters)
        self.onmtf.initialize(sqrt=False, initialization='clustering')

        self._update_best_model()

        start_time = time.time()
        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        validation_every_n=es_validation_every_n,
                                        stop_on_validation=es_stop_on_validation,
                                        lower_validations_allowed=es_lower_validations_allowed,
                                        evaluator_object=es_evaluator_object,
                                        validation_metric=es_validation_metric)
        print('')
        print('time: ' + str(time.time() - start_time))

        B_codebook = codebook_construction(self.URM_source, self.F_best, self.G_best, binarize=True)

        # Codebook transfer
        transfer = CodebookTransfer(self.URM_train, B_codebook, transfer_attempts=transfer_attempts, maximum_fill_iterations=maximum_fill_iterations)
        transfer.search_local_minimum()
        transfer.fill_matrix()
        self.transfer_loss = transfer.loss_best

        print('')
        print('FILLED TARGET MATRIX')
        print(transfer.URM_target_train_filled)

        self.URM_target_train_filled = sps.csr_matrix(transfer.URM_target_train_filled)
        self.URM_target_train_filled.eliminate_zeros()
        self.URM_target_train_filled = check_matrix(self.URM_target_train_filled, 'csr', dtype=np.float32)

        similarity = Compute_Similarity(self.URM_target_train_filled.T, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity, **similarity_parameters)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr', dtype=np.float32)

    def _run_epoch(self, currentEpoch):
        self._iteration_loss = self.onmtf.update()
        if currentEpoch % 100 == 0:
            print('construction iteration: ' + str(currentEpoch) + ', loss: ' + str(self._iteration_loss))

    def _prepare_model_for_validation(self):
        pass

    def _update_best_model(self):
        self.F_best = self.onmtf.F
        self.G_best = self.onmtf.G
        self.construction_loss = self.onmtf.loss

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        self._check_format()
        user_weights_array = self.W_sparse[user_id_array]
        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_target_train_filled.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_weights_array.dot(self.URM_target_train_filled).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_target_train_filled).toarray()
        return item_scores

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        data_dict_to_save = {"W_sparse": self.W_sparse, 'construction_loss': self.construction_loss, 'transfer_loss': self.transfer_loss}
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)
        self._print("Saving complete")
