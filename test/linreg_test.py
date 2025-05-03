from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sm_algo.linreg import LinearRegression


# Загрузка датасета diabetes из sklearn
data = load_diabetes()
X, y = data.data, data.target  # X - признаки, y - целевая переменная

# Разделение данных на обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # random_state для воспроизводимости
)

# Нормализация данных (приведение к нулевому среднему и единичной дисперсии)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # обучение scaler и трансформация тренировочных данных
X_test = scaler.transform(X_test)  # трансформация тестовых данных (без повторного обучения)

# Создание и обучение модели линейной регрессии
model = LinearRegression(learning_rate=0.1, epochs=1000)  # задаем скорость обучения и кол-во эпох
model.fit(X_train, y_train)  # обучение модели на тренировочных данных

# Получение предсказаний на тестовых данных
y_pred = model.predict(X_test)

# Оценка качества модели:
# - MSE (среднеквадратичная ошибка, чем меньше - тем лучше)
# - R² (коэффициент детерминации, 1 - идеальное предсказание)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# Вывод обученных весов модели
print("\nВеса модели:")
for i, w in enumerate(model.weights):
    print(f"Feature {i}: {w:.3f}")  # вес i-го признака
print(f"Intercept (bias): {model.bias:.3f}")  # свободный член (смещение)

# Для сравнения с реализацией линейной регрессии из sklearn
from sklearn.linear_model import LinearRegression as SklearnLR

sklearn_model = SklearnLR()  # создание модели sklearn
sklearn_model.fit(X_train, y_train)  # обучение
y_pred_sklearn = sklearn_model.predict(X_test)  # предсказание

# Вывод метрик sklearn модели для сравнения
print("\nSklearn Results:")
print(f"MSE: {mean_squared_error(y_test, y_pred_sklearn):.2f}")
print(f"R²: {r2_score(y_test, y_pred_sklearn):.2f}")