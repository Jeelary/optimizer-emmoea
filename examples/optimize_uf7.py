import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from optimizer import EMMOEA
from problems import UF7

def compute_igd(fitness, true_front):
    """
    Вычислить IGD (Inverted Generational Distance)
    Среднее расстояние от полученного фронта до истинного
    """
    if len(fitness) == 0:
        return np.inf
    
    distances = cdist(true_front, fitness)
    min_distances = np.min(distances, axis=1)
    igd = np.mean(min_distances)
    return igd


def run_example():   

    bounds = np.array([
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
    ])
    
    optimizer = EMMOEA(
        num_pop=100,
        num_obj=2,
        num_var=10,
        bounds=bounds,
        problem=UF7,
        surrogate='MultiTaskIBNN',
        max_evals=400,
        gmax=10
    )
    
    print("="*70)
    print("EMMOEA - Оптимизация UF7")
    print("="*70)
    print(f"Популяция:       {optimizer.N}")
    print(f"Целевых функций:      {optimizer.M}")
    print(f"Переменных:      {optimizer.D}")
    print(f"Вычислений: {optimizer.num_evals}")
    print("="*70)
    
    decisions, objectives = optimizer.optimize()
    x_true = np.linspace(0, 1, 1000).reshape(-1, 1)
    y_true = 1 - x_true
    true_front = np.hstack([x_true, y_true])
    igd = compute_igd(objectives, true_front)
    #objectives = UF7(decisions)
    plt.figure(figsize=(10, 6))
    plt.scatter(objectives[:, 0], objectives[:, 1], alpha=0.6, s=30, label=f'IGD={igd:.4f}')
    plt.plot(x_true, y_true, 'r--', label='Истинный фронт', linewidth=2)    
    plt.xlabel('f1', fontsize=12)
    plt.ylabel('f2', fontsize=12)
    plt.title('Фронт Парето - UF7', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('emmoea_result.png', dpi=150)
    print("\n✓ График сохранён как 'emmoea_result.png'")
    plt.show()    

if __name__ == "__main__":
    run_example()
