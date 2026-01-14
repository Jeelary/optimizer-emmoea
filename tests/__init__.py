"""
Тесты для EMMOEA
"""

import sys
import numpy as np
from problems import UF7, UF1, ZDT1
from emmoea import EMMOEA


def test_problem_uf7():
    """Тест функции UF7"""
    X = np.random.rand(5, 10)
    f = UF7(X)
    assert f.shape == (5, 2), f"Expected shape (5, 2), got {f.shape}"
    print("✓ UF7 test passed")


def test_problem_single_point():
    """Тест на одной точке"""
    X = np.random.rand(10)
    f = UF7(X)
    assert f.shape == (2,), f"Expected shape (2,), got {f.shape}"
    print("✓ Single point test passed")


def test_optimizer_initialization():
    """Тест инициализации оптимизатора"""
    bounds = np.array([[0, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    optimizer = EMMOEA(
        num_pop=50,
        num_obj=2,
        num_var=10,
        bounds=bounds,
        problem=UF7,
        surrogate='KRG',
        max_evals=100,
        gmax=5
    )
    assert optimizer.N == 50
    assert optimizer.M == 2
    assert optimizer.D == 10
    print("✓ Optimizer initialization test passed")


if __name__ == "__main__":
    print("Running EMMOEA tests...\n")
    test_problem_uf7()
    test_problem_single_point()
    test_optimizer_initialization()
    print("\n✓ All tests passed!")
