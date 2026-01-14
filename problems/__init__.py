"""
Тестовые функции оптимизации
"""

import numpy as np


def UF7(X):
    # Преобразуем вектор в матрицу с одной строкой, если нужно
    if len(X.shape) == 1:
        X = X.reshape(1, -1)

    n = X.shape[1]  # Количество столбцов
    j_indices = np.arange(1, n).reshape(1, -1)  # Индексы от 1 до n-1

    # Вычисляем sin_terms
    sin_terms = np.sin(6 * np.pi * X[:, [0]] + (j_indices + 1) * np.pi / n)

    # Определяем чётные и нечётные индексы (начиная с 3 для чётных и 2 для нечётных)
    J1 = np.arange(2, n, 2)  # Чётные индексы (начиная с 3)
    J2 = np.arange(1, n, 2)  # Нечётные индексы (начиная с 2)

    # Вычисляем разности
    Y = X[:, 1:] - sin_terms

    # Суммы квадратов для чётных и нечётных индексов
    sum1 = np.mean(Y[:, J1 - 1] ** 2, axis=1)  # Используем mean вместо sum
    sum2 = np.mean(Y[:, J2 - 1] ** 2, axis=1)

    # Вычисляем f1 и f2
    f1 = X[:, 0] ** 0.2 + 2 * sum1
    f2 = 1 - X[:, 0] ** 0.2 + 2 * sum2
    result = np.vstack((f1, f2)).T
    return result.squeeze()


def UF1(X):
    """
    Многоцелевая тестовая функция UF1
    2 целевые функции, 30 переменных
    """
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    n = X.shape[1]
    J1 = np.arange(2, n + 1, 2)
    J2 = np.arange(3, n + 1, 2)
    
    f1 = X[:, 0] + 2 * np.mean(X[:, J1 - 1] ** 2 - np.cos(4 * np.pi * X[:, J1 - 1]), axis=1)
    f2 = 1 - np.sqrt(X[:, 0]) + 2 * np.mean(X[:, J2 - 1] ** 2 - np.cos(4 * np.pi * X[:, J2 - 1]), axis=1)
    
    result = np.vstack((f1, f2)).T
    return result.squeeze()


def ZDT1(X):
    """
    Классическая тестовая функция ZDT1
    2 целевые функции, 30 переменных
    """
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    f1 = X[:, 0]
    g = 1 + 9 * np.mean(X[:, 1:], axis=1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    
    result = np.vstack((f1, f2)).T
    return result.squeeze()
