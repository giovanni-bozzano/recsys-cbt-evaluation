import numpy as np
from numpy.linalg import multi_dot


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
        self.prediction = None
        self.URM_target_train_filled = None

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
            loss = np.ndarray(self.k)
            for j in range(self.k):
                loss[j] = np.linalg.norm((self.URM_target_train[i, :] - BV_t[j, :]) * self.W[i, :])
            j = np.argmin(loss)
            self.U[i, :] = 0.0
            self.U[i, j] = 1.0

    def _update_V(self):
        UB = np.dot(self.U, self.B)
        for i in range(self.q):
            loss = np.ndarray(self.l)
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
        self.prediction = self.W_flipped * multi_dot([self.U_best, self.B, self.V_best.T])
        self.URM_target_train_filled = self.W * self.URM_target_train + self.prediction
        # self.URM_target_train_filled = self.W * self.URM_target_train + self.W_flipped * np.ones((self.p, self.q))
