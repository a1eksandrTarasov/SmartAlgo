from sklearn.datasets import load_iris
from sm_algo.kmeans import KMeans
import matplotlib.pyplot as plt


# Загрузка данных Iris (150 samples, 4 features)
iris = load_iris()
X = iris.data  # Данные в виде матрицы [150x4]

# Создание модели с 3 кластерами (по числу видов ирисов)
kmeans = KMeans(n_clusters=3)

# Обучение модели на масштабированных данных
kmeans.fit(X)

# Получение меток кластеров для каждой точки
labels = kmeans.labels_

# Рисуем точки по первым двум признакам, раскрашивая по кластерам
plt.scatter(
    X[:, 0],  # Ось X: первый признак (длина чашелистика)
    X[:, 1],  # Ось Y: второй признак (ширина чашелистика)
    c=labels,        # Цвет точек = номер кластера
    cmap='viridis'   # Палитра цветов
)

# Рисуем центроиды
plt.scatter(
    kmeans.centroids[:, 0],  # X координата центроидов
    kmeans.centroids[:, 1],  # Y координата центроидов
    marker='X',              # Форма — крестик
    s=200,                   # Размер
    c='red',                 # Цвет
    edgecolor='black'        # Обводка
)

# Подписи осей и заголовок
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.title('K-means Clustering on Iris Dataset')
plt.show()