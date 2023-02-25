import numpy as np
import scipy.sparse as sps
from numpy.linalg import multi_dot
from pyfm import pylibfm


class CodebookTransfer:
    def __init__(self, URM_target_train, B, transfer_attempts, maximum_fill_iterations):
        self.URM_target_train = URM_target_train.copy().toarray()  # target matrix to fill
        self.B = B  # codebook
        self.p, self.q = URM_target_train.shape  # n_users, n_items
        self.k, self.l = B.shape  # n_user_clusters, n_item_clusters
        self.transfer_attempts = transfer_attempts
        self.maximum_fill_iterations = maximum_fill_iterations
        self.W = None
        self.W_flipped = None
        self.U = None
        self.V = None
        self.U_best = None
        self.V_best = None
        self.loss_best = None
        self.prediction = np.empty((self.URM_target_train.shape[0] * self.URM_target_train.shape[1] - np.count_nonzero(self.URM_target_train)))
        self.URM_target_train_filled = URM_target_train.copy().toarray().flatten()

    def _generate_W(self):
        self.W = np.zeros((self.p, self.q))
        self.W = np.divide(self.URM_target_train, self.URM_target_train)
        self.W = np.nan_to_num(self.W)
        self.W_flipped = 1 - self.W

    def _initialize_U(self):
        self.U = np.zeros((self.p, self.k))

    def _initialize_V(self):
        self.V = np.zeros((self.q, self.l))
        for i in range(self.q):
            j = np.random.randint(self.l)
            self.V[i, j] = 1.0

    def _update_U(self):
        BV_t = np.dot(self.B, self.V.T)
        for i in range(self.p):
            loss = np.empty(self.k)
            for j in range(self.k):
                loss[j] = np.linalg.norm((self.URM_target_train[i, :] - BV_t[j, :]) * self.W[i, :])
            j = np.argmin(loss)
            self.U[i, :] = 0.0
            self.U[i, j] = 1.0

    def _update_V(self):
        UB = np.dot(self.U, self.B)
        for i in range(self.q):
            loss = np.empty(self.l)
            for j in range(self.l):
                loss[j] = np.linalg.norm((self.URM_target_train[:, i] - UB[:, j]) * self.W[:, i])
            j = np.argmin(loss)
            self.V[i, :] = 0.0
            self.V[i, j] = 1.0

    def search_local_minimum(self):
        self._generate_W()

        print('')
        self.loss_best = np.inf
        for attempt in range(self.transfer_attempts):
            print('== transfer attempt: ' + str(attempt + 1) + '/' + str(self.transfer_attempts))

            self._initialize_U()
            self._initialize_V()
            iteration_loss_best = np.inf

            for i in range(self.maximum_fill_iterations):
                self._update_U()
                self._update_V()
                # Remove the filled values from the generated target matrix and compare it to the original one
                loss = np.linalg.norm((self.URM_target_train - multi_dot([self.U, self.B, self.V.T])) * self.W)
                print('loss: ' + str(loss))
                # The loss function is monotone, so stop if we have reached the local minimum
                if loss == iteration_loss_best:
                    break
                iteration_loss_best = loss

            # Generate multiple local minima and keep the best one
            if iteration_loss_best < self.loss_best:
                self.loss_best = iteration_loss_best
                self.U_best = self.U
                self.V_best = self.V

    def fill_matrix(self):
        train = sps.lil_matrix((np.count_nonzero(self.URM_target_train), self.p + self.q + self.B.shape[0] + self.B.shape[1] + 1), dtype=np.double)
        validation = np.empty((np.count_nonzero(self.URM_target_train)))
        test = sps.lil_matrix((self.URM_target_train.shape[0] * self.URM_target_train.shape[1] - np.count_nonzero(self.URM_target_train), self.p + self.q + self.B.shape[0] + self.B.shape[1] + 1),
                              dtype=np.double)
        test_counter = 0
        train_counter = 0
        user_cluster_start = self.p + self.q
        user_cluster_end = self.p + self.q + self.B.shape[0]
        item_cluster_end = self.p + self.q + self.B.shape[0] + self.B.shape[1]

        for user_index in range(self.p):
            for item_index in range(self.q):
                codebook_value = self.B[np.where(self.U_best[user_index] == 1), np.where(self.V_best[item_index] == 1)]
                if self.URM_target_train[user_index, item_index] != 0:
                    train[train_counter, user_index] = 1
                    train[train_counter, self.p + item_index] = 1
                    train[train_counter, user_cluster_start:user_cluster_end] = self.U_best[user_index]
                    train[train_counter, user_cluster_end:item_cluster_end] = self.V_best[item_index]
                    train[train_counter, item_cluster_end] = codebook_value
                    validation[train_counter] = np.double(self.URM_target_train[user_index, item_index])
                    train_counter += 1
                else:
                    test[test_counter, user_index] = 1
                    test[test_counter, self.p + item_index] = 1
                    test[test_counter, user_cluster_start:user_cluster_end] = self.U_best[user_index]
                    test[test_counter, user_cluster_end:item_cluster_end] = self.V_best[item_index]
                    test[test_counter, item_cluster_end] = codebook_value
                    test_counter += 1
        train = train.tocsr()
        test = test.tocsr()

        fm = pylibfm.FM(num_factors=50, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001)
        fm.fit(train, validation)

        iterations = np.math.ceil(len(self.prediction) / 10.0)
        for iteration in range(iterations):
            if iteration != iterations - 1:
                self.prediction[iteration * 10:(iteration + 1) * 10] = fm.predict(test[iteration * 10:(iteration + 1) * 10])
            else:
                self.prediction[iteration * 10:] = fm.predict(test[iteration * 10:])

        self.URM_target_train_filled[self.URM_target_train_filled == 0] = self.prediction
        self.URM_target_train_filled = np.reshape(self.URM_target_train_filled, (-1, self.URM_target_train.shape[1]))
