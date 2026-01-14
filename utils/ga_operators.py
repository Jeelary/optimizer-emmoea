"""
Генетические операторы
"""

import numpy as np


def GAreal(Population, bounds, proM=1, proC=1, disC=20, disM=20):
    lower = bounds[0]
    upper = bounds[1]
    """
    Генетические операторы для вещественных переменных:
    - Симулированный бинарный кроссовер (SBX)
    - Полиномиальная мутация
    """
    N, _ = Population.shape
    if N % 2 != 0:
        Population = Population[:-1]
        N -= 1
    mid_index = N // 2
    Parent1 = Population[:mid_index]
    Parent2 = Population[mid_index:]

    N, D = Parent1.shape

    # --- SBX (Simulated Binary Crossover) ---
    
    mu = np.random.rand(N, D)
    r1 = np.random.randint(0, 2, size=(N, D))
    r2 = np.random.rand(N, D)
    r3 = np.random.rand(N, 1)
    beta = np.zeros((N, D))
    mask_low = mu <= 0.5
    beta[mask_low] = (2 * mu[mask_low]) ** (1 / (disC + 1))
    mask_high = mu > 0.5
    beta[mask_high] = (2 - 2 * mu[mask_high]) ** (-1 / (disC + 1))
    beta = beta * ((-1) ** r1)
    beta[r2 < 0.5] = 1
    mask_r3 = np.repeat(r3 > proC, D, axis=1)
    beta[mask_r3] = 1
    Offspring = np.vstack((
        (Parent1 + Parent2) / 2 + beta * (Parent1 - Parent2) / 2,
        (Parent1 + Parent2) / 2 - beta * (Parent1 - Parent2) / 2
    ))

    # --- Polynomial mutation ---
    
    Lower = np.tile(lower, (2 * N, 1))
    Upper = np.tile(upper, (2 * N, 1))
    r4 = np.random.rand(2 * N, D)
    mu = np.random.rand(2 * N, D)
    Site = r4 < proM / D
    temp1 = Site & (mu <= 0.5)
    temp2 = Site & (mu > 0.5)
    Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
    delta = (Offspring[temp1] - Lower[temp1]) / (Upper[temp1] - Lower[temp1])
    factor = (2 * mu[temp1] + (1 - 2 * mu[temp1]) * (1 - delta) ** (disM + 1)) ** (1 / (disM + 1)) - 1
    Offspring[temp1] = Offspring[temp1] + (Upper[temp1] - Lower[temp1]) * factor
    delta = (Upper[temp2] - Offspring[temp2]) / (Upper[temp2] - Lower[temp2])
    factor = 1 - (2 * (1 - mu[temp2]) + 2 * (mu[temp2] - 0.5) * (1 - delta) ** (disM + 1)) ** (1 / (disM + 1))
    Offspring[temp2] = Offspring[temp2] + (Upper[temp2] - Lower[temp2]) * factor

    return Offspring
