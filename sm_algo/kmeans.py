import numpy as np


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        indices = np.random.permutation(X.shape[0])
        self.centroids = X[indices[:self.n_clusters]]

    def _compute_distances(self, X):
        # Векторизованный расчет евклидовых расстояний
        return np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))

    def fit(self, X):
        self._initialize_centroids(X)

        for _ in range(self.max_iter):
            # Расчет расстояний и назначение кластеров
            distances = self._compute_distances(X)
            self.labels_ = np.argmin(distances, axis=1)

            # Обновление центроидов
            new_centroids = np.array([
                X[self.labels_ == i].mean(axis=0)
                for i in range(self.n_clusters)
            ])

            # Проверка сходимости
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break

            self.centroids = new_centroids

        return self

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def get_centroids(self):
        """Возвращает координаты финальных центроидов"""
        return self.centroids.copy()