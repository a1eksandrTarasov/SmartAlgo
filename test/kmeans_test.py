import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sm_algo.kmeans import KMeans

# Загрузка данных
iris = load_iris()
X = iris.data

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Кластеризация
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)  # 1. Вызываем fit()
labels = kmeans.labels_  # 2. Получаем метки

# Визуализация (первые 2 признака)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')

# 3. Исправленный вызов get_centroids()
centroids = kmeans.get_centroids()
plt.scatter(centroids[:, 0], centroids[:, 1],
           marker='X', s=200, c='red', edgecolor='black')

plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.title('K-means Clustering on Iris Dataset')
plt.show()