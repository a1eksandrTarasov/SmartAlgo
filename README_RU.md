# SmartAlgo: Облегченная библиотека машинного обучения  

`SmartAlgo` — это Python-пакет с реализацией основных алгоритмов машинного обучения "с нуля". Разработан для простоты и обучения, включает линейную регрессию, k-средних и логистическую регрессию с L2-регуляризацией. Работает с минимальными зависимостями (`numpy` и `scipy`), идеально подходит для изучения основ ML.  

## Установка  

Установите пакет через pip:  

```bash  
pip install sm_algo  
```  

**Зависимости:**  
- numpy
- scipy  

## Алгоритмы  

### 1. Линейная регрессия с градиентным спуском  
**Класс:** `LinearRegression`  
Линейная регрессия, обученная методом градиентного спуска.  

**Параметры:**  
- `learning_rate` (float, по умолчанию=0.01): Шаг градиентного спуска.  
- `epochs` (int, по умолчанию=1000): Количество итераций обучения.  

**Методы:**  
- `fit(X, y)`: Обучает модель на данных `X` и целевых значениях `y`.  
- `predict(X)`: Возвращает предсказания для входных данных `X`.  

**Пример:**  
```python  
from sm_algo.linreg import LinearRegression  
from sklearn.datasets import load_diabetes  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  

# Загрузка данных  
data = load_diabetes()  
X, y = data.data, data.target  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Нормализация данных  
scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)  

# Обучение модели  
model = LinearRegression(learning_rate=0.1, epochs=1000)  
model.fit(X_train, y_train)  

# Предсказание и оценка  
y_pred = model.predict(X_test)  
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")  
print(f"R²: {r2_score(y_test, y_pred):.2f}")  
```  

### 2. Кластеризация K-средних  
**Класс:** `KMeans`  
Алгоритм K-средних с инициализацией центроидов методом k-means++.  

**Параметры:**  
- `n_clusters` (int, по умолчанию=8): Число кластеров.  
- `max_iter` (int, по умолчанию=300): Максимальное число итераций.  
- `tol` (float, по умолчанию=1e-4): Допуск для определения сходимости.  
- `random_state` (int, опционально): Seed для инициализации центроидов.  

**Методы:**  
- `fit(X)`: Выполняет кластеризацию для данных `X`.  
- `predict(X)`: Возвращает индексы кластеров для новых данных.  

**Пример:**  
```python  
from sm_algo.kmeans import KMeans  
from sklearn.datasets import load_iris  
import matplotlib.pyplot as plt  

# Загрузка данных  
iris = load_iris()  
X = iris.data  

# Кластеризация  
kmeans = KMeans(n_clusters=3)  
kmeans.fit(X)  
labels = kmeans.labels_  

# Визуализация (по двум признакам)  
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')  
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='X', s=200, c='red')  
plt.xlabel('Длина чашелистика')  
plt.ylabel('Ширина чашелистика')  
plt.show()  
```  

### 3. Логистическая регрессия с L2-регуляризацией  
**Класс:** `LogisticRegressionRidge`  
Логистическая регрессия с L2-регуляризацией, обучение градиентным спуском.  

**Параметры:**  
- `learning_rate` (float, по умолчанию=0.01): Шаг градиентного спуска.  
- `lambda_` (float, по умолчанию=0.1): Сила L2-регуляризации.  
- `epochs` (int, по умолчанию=1000): Число итераций обучения.  
- `fit_intercept` (bool, по умолчанию=True): Добавлять ли свободный член.  
- `verbose` (bool, по умолчанию=False): Вывод лога потерь каждые 100 эпох.  

**Методы:**  
- `fit(X, y)`: Обучает модель.  
- `predict_proba(X)`: Возвращает вероятности классов.  
- `predict(X, threshold=0.5)`: Возвращает метки классов.  

**Пример:**  
```python  
from sm_algo.logisticreg import LogisticRegressionRidge  
from sklearn.datasets import make_classification  
from sklearn.model_selection import train_test_split  

# Генерация синтетических данных  
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, random_state=42)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Обучение модели  
model = LogisticRegressionRidge(learning_rate=0.001, lambda_=0.1, epochs=1000)  
model.fit(X_train, y_train)  

# Предсказание  
y_pred = model.predict(X_test)  
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")  
```