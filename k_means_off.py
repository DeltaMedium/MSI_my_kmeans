import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

"DOMYŚLNE WARTOŚCI TO 3 KLASTRY, MINIMALNA ITERACJA 10, MAKSYMALNA ITERACJA 500 <- MOŻNA JE ZMIENIĆ WYWOŁUJĄC FUNKCJE"


class my_kmeans_msi(TransformerMixin, ClusterMixin, BaseEstimator):
    @_deprecate_positional_args
    def __init__(self, n_clusters=5, max_iter=500):
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        """if (n_clusters<1):
            "jest to zabezpieczenie przeciwko sytuacji w której będziemy mieć mniej niż jeden klaseter"
            print("Liczba klastrow zmieniona na domyslna wartosc 5")
            k = 5"""

    # Z uwagi na fakt, iż kmeans jest algorytmem bez nadzorcy nie ma konieczności przekazywania do niego zbioru treningowego
    def fit(self, X, y=None):
        # X, ys = check_X_y(X, y)
        samples = X.shape[0]  # liczba próbek
        classifications = X.shape[1]  # liczba cech/atrybutów/właściwości - zależnie od badanych danych
        c_mean = np.mean(X, 0)
        std = np.std(X, 0)
        f_centers_random = np.random.randn(self.n_clusters, classifications) * std + c_mean
        """Tworzenie buforów współrzędnych centroidów - wartość początkowa to skopiowana wartość zwracana przez powyższy randn"""
        buffer_of_centroids = f_centers_random
        buffer_of_centroids_2 = f_centers_random
        matrix_of_clusters = []  # matryca zawierająca informacje o tym do jakiego klastra należy próbka
        matrix_of_distances = np.zeros((samples, self.n_clusters))

        for kmeans_round in range(self.max_iter):
            buffer_of_centroids_2 = buffer_of_centroids
            for i in range(samples):
                for j in range(self.n_clusters):
                    """Tworzenie tablicy z dystansami do każdego środka centroidu"""
                    matrix_of_distances[i, j] = distance.euclidean(X[i], buffer_of_centroids[j])
                matrix_of_clusters = np.argmin(matrix_of_distances, 1)

            for i in range(self.n_clusters):
                "Obliczanie nowych środków centroidów "
                bool_check_k = False
                for j in range(samples):
                    if (matrix_of_clusters[j] == i):
                        bool_check_k = True
                if (bool_check_k):
                    buffer_of_centroids[i] = np.mean(X[matrix_of_clusters == i], 0)

            if (self.check_tables(buffer_of_centroids, buffer_of_centroids_2)):
                break
        self.matrix_of_clusters = matrix_of_clusters
        return self

    def check_tables(self, a, b):
        ia = a.shape[0]
        ja = a.shape[1]
        tim = 0
        for i in range(ia):
            for j in range(ja):
                if (a[i, j] == b[i, j]):
                    tim += 1
        if ((ia * ja) == tim):
            return True
        else:
            return False

    def fit_predict(self, X):
        return self.fit(X).matrix_of_clusters
