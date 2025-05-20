from sm_algo.simplex import SimplexMethod
import numpy as np
import unittest


class TestSimplexMethod(unittest.TestCase):
    def test_simple_problem(self):
        """Тест на простой задаче с очевидным решением.
        Проверяет, что метод корректно решает задачу с двумя переменными
        и двумя ограничениями, находит оптимальное значение и верное решение."""
        c = [3, 2]
        A = [
            [1, 1],
            [1, -1]
        ]
        b = [4, 2]

        simplex = SimplexMethod(c, A, b)
        result = simplex.solve()

        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['value'], 12.0)
        np.testing.assert_array_almost_equal(result['x'], [4, 0])

    def test_unbounded_problem(self):
        """Тест на неограниченную задачу.
        Проверяет, что метод корректно определяет, когда целевая функция
        может расти бесконечно в допустимой области."""
        c = [1, 1]
        A = [
            [-1, 1],
            [-1, 0]
        ]
        b = [1, 0]

        simplex = SimplexMethod(c, A, b)
        result = simplex.solve()

        self.assertFalse(result['success'])
        self.assertEqual(result['message'], 'Problem is unbounded')

    def test_multiple_solutions(self):
        """Тест на задачу с множеством решений.
        Проверяет, что метод находит одно из возможных решений,
        удовлетворяющее всем ограничениям."""
        c = [1, 1]
        A = [
            [1, 1]
        ]
        b = [2]

        simplex = SimplexMethod(c, A, b)
        result = simplex.solve()

        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['value'], 2.0)
        x1, x2 = result['x']
        self.assertGreaterEqual(x1, 0)  # Проверка неотрицательности
        self.assertGreaterEqual(x2, 0)
        self.assertLessEqual(x1 + x2, 2 + 1e-10)  # Проверка ограничения с учетом погрешности

    def test_degenerate_nonzero_solution(self):
        """Тест на вырожденную задачу с линейно зависимыми ограничениями.
        Проверяет, что метод корректно обрабатывает вырожденные случаи
        и находит ненулевое решение."""
        c = [1, 2]
        A = [
            [1, 1],
            [2, 2]  # Линейно зависимое ограничение
        ]
        b = [1, 2]

        simplex = SimplexMethod(c, A, b)
        result = simplex.solve()

        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['value'], 2.0)
        self.assertTrue(np.any(result['x'] > 0))  # Проверка, что хотя бы одна переменная > 0

    def test_highly_degenerate_problem(self):
        """Тест на сильно вырожденную задачу.
        Проверяет обработку случая, когда все ограничения фактически нулевые
        и тривиальное решение x=0 является оптимальным."""
        c = [1]
        A = [
            [1],
            [1],  # Три одинаковых ограничения
            [1]
        ]
        b = [0, 0, 0]

        simplex = SimplexMethod(c, A, b)
        result = simplex.solve()

        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['value'], 0.0)
        self.assertAlmostEqual(result['x'][0], 0.0)  # Проверка тривиального решения


if __name__ == '__main__':
    unittest.main()