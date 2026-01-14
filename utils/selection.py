"""
Методы отбора решений
"""

import numpy as np
from scipy.spatial.distance import cdist


def kriging_selection(PopObj, V):
    """
    Функция для отбора решений для следующего поколения
    Параметры:
        PopObj (np.ndarray): Матрица значений целевых функций для текущего набора решений.
        V (np.ndarray): Матрица референсных векторов.

    Возвращает:
        index (np.ndarray): Индексы выбранных решений для следующего поколения.
    """

    min_v = np.min(PopObj, axis=0)
    max_v = np.max(PopObj, axis=0)
    NormPopObj = (PopObj - min_v) / (max_v - min_v)
    # print('NormPopObj(std):', np.std(NormPopObj))
    cosine_distance = 1 - cdist(NormPopObj, V, metric='cosine')
    # print('V(std):', np.std(V))
    # print('cosine_distance(std):', np.std(cosine_distance))
    Angle = np.arccos(cosine_distance)
    # print('Angle(std):', np.std(Angle))
    associate = np.argmin(Angle, axis=1)
    # print('associate(mean):', np.mean(associate))

    NV = V.shape[0]
    Next = np.ones(NV, dtype=int) * -1

    for i in np.unique(associate):
        current = np.where(associate == i)[0]
        de = np.linalg.norm(NormPopObj[current, :], axis=1)
        best_index = np.argmin(de)
        Next[i] = current[best_index]

    index = Next[Next != -1]
    return index
