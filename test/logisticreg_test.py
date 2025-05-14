from sm_algo.logisticreg import LogisticRegressionRidge
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def score(model, X, y, threshold=0.5):
    """
    Вычисляет точность (accuracy) модели.

    Параметры:
    ----------
    X : ndarray, shape (n_samples, n_features)
        Матрица признаков.
    y : ndarray, shape (n_samples,)
        Истинные метки классов.
    threshold : float, default=0.5
        Порог классификации.

    Возвращает:
    -----------
    float
        Точность модели (accuracy).
    """
    y_pred = model.predict(X, threshold)
    return accuracy_score(y, y_pred)


def plot_loss_history(loss_history):
    """Визуализирует историю потерь во время обучения."""
    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


def plot_decision_boundary(model, X, y):
    """
    Визуализирует разделяющую границу (работает только для 2D-данных).

    Параметры:
    ----------
    X : ndarray, shape (n_samples, 2)
        Матрица признаков (только 2 признака).
    y : ndarray, shape (n_samples,)
        Метки классов.
    """
    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary работает только для 2D-данных!")

    # Сетка для визуализации
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Предсказание для сетки
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Визуализация
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title("Decision Boundary")
    plt.show()

X, y = make_classification(
        n_samples=10000,
        n_features=2,
        n_classes=2,
        n_redundant=0,
        class_sep=2.0,
        flip_y=0.1,
        random_state=42
    )

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model = LogisticRegressionRidge(learning_rate=0.001, lambda_=0.1, verbose=True, epochs=1000)
model.fit(X_train, y_train)

# Оценка модели
print("Веса модели:", model.get_weights())
train_accuracy = score(model, X_train, y_train)
test_accuracy = score(model, X_test, y_test)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Визуализация
plot_loss_history(model.loss_history)
plot_decision_boundary(model, X_test, y_test)