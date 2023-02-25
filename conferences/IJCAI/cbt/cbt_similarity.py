import numpy as np

from Base.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender
from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
from Base.Recommender_utils import check_matrix
from Base.Similarity.Compute_Similarity import Compute_Similarity


class CBTSimilarityRecommender(BaseUserSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "CBTSimilarityRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, URM_train_filled, verbose=True):
        self.URM_train_filled = URM_train_filled
        super(CBTSimilarityRecommender, self).__init__(URM_train, verbose=verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting="none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.URM_train_filled = self.URM_train_filled.astype(np.float32)
            self.URM_train_filled = okapi_BM_25(self.URM_train_filled.T).T
            self.URM_train_filled = check_matrix(self.URM_train_filled, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train_filled = self.URM_train_filled.astype(np.float32)
            self.URM_train_filled = TF_IDF(self.URM_train_filled.T).T
            self.URM_train_filled = check_matrix(self.URM_train_filled, 'csr')

        similarity = Compute_Similarity(self.URM_train_filled.T, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        self._check_format()

        user_weights_array = self.W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train_filled.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_weights_array.dot(self.URM_train_filled).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_train_filled).toarray()

        return item_scores
