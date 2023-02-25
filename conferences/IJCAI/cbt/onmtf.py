"""
Implemented based on
Gang Chen, Fei Wang, and Changshui Zhang. (2007).
Collaborative Filtering Using Orthogonal Nonnegative Matrix Tri-factorization.
Proceedings of the IEEE Int’l Conf. on Data Mining Workshops.
2007. 303–308. 10.1109/ICDMW.2007.18.
"""
import numpy as np
from numpy.linalg import multi_dot
from sklearn.cluster import KMeans


class ONMTF(object):
    def __init__(self, X, k, l):
        self.X = X.copy()
        self.m, self.n = X.shape
        self.k = k
        self.l = l
        self.sqrt = None
        self.F = None
        self.G = None
        self.S = None
        self.loss = None

    def _loss(self):
        """
        ||X-FSGt||^2, euclidean error
        """
        FSGt = multi_dot([self.F, self.S, self.G.T])
        self.loss = np.linalg.norm(self.X - FSGt)  # frobenius norm
        # loss = np.linalg.norm(self.X - FSGt, ord=2)  # ord 2 norm
        # loss = np.sqrt(np.mean((self.X - FSGt) ** 2))  # RMSE
        return self.loss

    def _update_G(self):
        enum = multi_dot([self.X.T, self.F, self.S])
        denom = multi_dot([self.G, self.G.T, self.X.T, self.F, self.S])
        if self.sqrt:
            self.G *= np.nan_to_num(np.sqrt(enum / denom))
        else:
            self.G *= np.nan_to_num(enum / denom)

    def _update_F(self):
        enum = multi_dot([self.X, self.G, self.S.T])
        denom = multi_dot([self.F, self.F.T, self.X, self.G, self.S.T])
        if self.sqrt:
            self.F *= np.nan_to_num(np.sqrt(enum / denom))
        else:
            self.F *= np.nan_to_num(enum / denom)

    def _update_S(self):
        enum = multi_dot([self.F.T, self.X, self.G])
        denom = multi_dot([self.F.T, self.F, self.S, self.G.T, self.G])
        if self.sqrt:
            self.S *= np.nan_to_num(np.sqrt(enum / denom))
        else:
            self.S *= np.nan_to_num(enum / denom)

    def cluster(self, X, k):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = set(kmeans.labels_)
        labeled_features = kmeans.labels_
        return np.array([np.multiply([i == k for i in labeled_features], 1) for k in labels]).T.astype(np.float64)

    def _initialize_clustering(self):
        """
        Initialize F, G, and S latent matrices according to Ding et al. 2006:
        1. G is obtained via k means clustering of columns of X. G = G+0.2
        2. F is obtained via k means clustering of rows of X. F = F+ 0.2
        3. S is obtained via S = F.T@X@G
        """
        # Initialization based on Ding et al 2006 Sec.5
        self.G = self.cluster(self.X.T, self.l)
        self.G += 0.2
        self.F = self.cluster(self.X, self.k)
        self.F += 0.2
        self.S = multi_dot([self.F.T, self.X, self.G])

    def _initialize_random(self):
        self.F = np.random.rand(self.m, self.k).astype(np.float64)
        self.G = np.random.rand(self.n, self.l).astype(np.float64)
        self.S = np.random.rand(self.k, self.l).astype(np.float64)

    def initialize(self, sqrt, initialization):
        self.sqrt = sqrt
        if initialization == 'clustering':
            self._initialize_clustering()
        else:
            self._initialize_random()

    def update(self):
        self._update_G()
        self._update_F()
        self._update_S()

        return self._loss()
