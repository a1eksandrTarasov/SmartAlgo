from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sm_algo.linreg import LinearRegression


# Загрузка данных
data = load_diabetes()
X, y = data.data, data.target

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение модели
model = LinearRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)

# Оценка модели
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")
print("\nВеса модели:")
for i, w in enumerate(model.weights):
    print(f"Feature {i}: {w:.3f}")
print(f"Intercept (bias): {model.bias:.3f}")

# Для сравнения с sklearn
from sklearn.linear_model import LinearRegression as SklearnLR

sklearn_model = SklearnLR()
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)

print("\nSklearn Results:")
print(f"MSE: {mean_squared_error(y_test, y_pred_sklearn):.2f}")
print(f"R²: {r2_score(y_test, y_pred_sklearn):.2f}")