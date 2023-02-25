import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'framework'))

import numpy as np
import scipy.sparse as sps

from framework.Base.Recommender_utils import check_matrix
from conferences.ECML.lkt_fm.lkt_fm import LKTFMRecommender
from conferences.IJCAI.cbt.data.parse_data import fill_row_miss_value
from conferences.IJCAI.cbt.evaluator_loss_wrapper import EvaluatorLossWrapper

URM_source = np.array([[2, 3, 3, 3, 2, 0],
                       [3, 1, 2, 2, 0, 1],
                       [3, 1, 0, 2, 3, 1],
                       [0, 2, 1, 1, 3, 2],
                       [2, 3, 3, 3, 2, 3],
                       [3, 2, 1, 0, 3, 2]], dtype=np.float32)
URM_source = sps.csr_matrix(URM_source)
URM_source.eliminate_zeros()
URM_source = check_matrix(URM_source, 'csr', dtype=np.float32)
URM_source = fill_row_miss_value(URM_source)
URM_source = sps.csr_matrix(URM_source)
URM_source.eliminate_zeros()
URM_source = check_matrix(URM_source, 'csr', dtype=np.float32)

URM_target = np.array([[0, 1, 3, 3, 1],
                       [3, 3, 2, 0, 3],
                       [2, 2, 0, 3, 0],
                       [1, 1, 3, 0, 0],
                       [1, 0, 0, 3, 1],
                       [3, 0, 2, 2, 3],
                       [0, 2, 3, 3, 2]], dtype=np.float32)
URM_target_filled_expected = np.array([[1, 1, 3, 3, 1],
                                       [3, 3, 2, 2, 3],
                                       [2, 2, 3, 3, 2],
                                       [1, 1, 3, 3, 1],
                                       [1, 1, 3, 3, 1],
                                       [3, 3, 2, 2, 3],
                                       [2, 2, 3, 3, 2]], dtype=np.float32)
URM_target = sps.csr_matrix(URM_target)
URM_target.eliminate_zeros()
URM_target = check_matrix(URM_target, 'csr', dtype=np.float32)

cbt = LKTFMRecommender(URM_target_train=URM_target, URM_source=URM_source)
cbt.fit(n_clusters=3, es_evaluator_object=EvaluatorLossWrapper())
# cbt = CBTRecommender(URM_target_train=URM_target, URM_source=URM_source, baseline=False)
# cbt.fit(3, 3, es_evaluator_object=EvaluatorLossWrapper())
print('')
print('EXPECTED FILLED TARGET MATRIX')
print(URM_target_filled_expected)
URM_target_filled = np.rint(cbt.URM_target_train_filled.toarray())
print('')
print('MATCH: ' + str((URM_target_filled == URM_target_filled_expected).all()))
print(URM_target_filled == URM_target_filled_expected)
