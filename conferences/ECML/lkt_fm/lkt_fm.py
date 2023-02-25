import time

import numpy as np
import scipy.sparse as sps
from sklearn.cluster import KMeans

from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
from Base.DataIO import DataIO
from Base.Recommender_utils import check_matrix
from Base.Similarity.Compute_Similarity import Compute_Similarity
from conferences.ECML.lkt_fm.mf import MF
from conferences.ECML.lkt_fm.codebook_transfer import CodebookTransfer
from conferences.IJCAI.cbt.codebook_construction import codebook_construction


class LKTFMRecommender(BaseSimilarityMatrixRecommender):
    _iteration_loss = None

    RECOMMENDER_NAME = 'LKTFMRecommender'

    def __init__(self, URM_target_train, URM_source, verbose=True):
        super(LKTFMRecommender, self).__init__(URM_target_train, verbose)

        self.URM_source = check_matrix(URM_source.copy(), 'csr', dtype=np.float32)

        self.construction_loss = None
        self.transfer_loss = None
        self.URM_target_train_filled = None
        self.W_sparse = None

    def fit(self,
            n_clusters=50,
            epochs=10,  # maximum_construct_iterations
            transfer_attempts=30,
            maximum_fill_iterations=100,

            # early-stopping parameters
            es_validation_every_n=1,
            es_stop_on_validation=False,
            es_lower_validations_allowed=2000,
            es_evaluator_object=None,
            es_validation_metric='loss',

            topK=20,
            shrink=0,
            similarity='pearson',
            normalize=False,
            feature_weighting='none',
            **similarity_parameters):
        # Codebook construction
        start_time = time.time()

        # model = LightFM(loss='logistic', learning_rate=0.001)
        # model.fit(self.URM_source, epochs=epochs, num_threads=2, verbose=True)
        model = MF(self.URM_source.toarray(), n_clusters, n_clusters, 0.1, 1, epochs)
        model.train()

        print('')
        print('time: ' + str(time.time() - start_time))

        F = self.cluster(model.P, n_clusters)
        G = self.cluster(model.Q, n_clusters)

        self.URM_source = self.URM_source.toarray()

        B_codebook = codebook_construction(self.URM_source, F, G, binarize=True)

        # Codebook transfer
        transfer = CodebookTransfer(self.URM_train,
                                    B_codebook,
                                    transfer_attempts=transfer_attempts,
                                    maximum_fill_iterations=maximum_fill_iterations)
        transfer.search_local_minimum()
        transfer.fill_matrix()
        self.transfer_loss = transfer.loss_best

        self.URM_target_train_filled = sps.csr_matrix(transfer.URM_target_train_filled)
        self.URM_target_train_filled.eliminate_zeros()
        self.URM_target_train_filled = check_matrix(self.URM_target_train_filled, 'csr', dtype=np.float32)

        similarity = Compute_Similarity(self.URM_target_train_filled.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_parameters)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr', dtype=np.float32)

    def cluster(self, X, k):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = set(kmeans.labels_)
        labeled_features = kmeans.labels_
        return np.array([np.multiply([i == k for i in labeled_features], 1) for k in labels]).T.astype(np.float64)

    def _prepare_model_for_validation(self):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        self._check_format()
        user_weights_array = self.W_sparse[user_id_array]
        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_target_train_filled.shape[1]),
                                    dtype=np.float32) * np.inf
            item_scores_all = user_weights_array.dot(self.URM_target_train_filled).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_target_train_filled).toarray()
        return item_scores

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        data_dict_to_save = {"W_sparse": self.W_sparse, 'construction_loss': self.construction_loss,
                             'transfer_loss': self.transfer_loss}
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)
        self._print("Saving complete")
