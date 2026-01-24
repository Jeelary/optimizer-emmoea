import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
from optimizer import EMMOEA
from problems import UF7
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


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


def run_optimization_thread(thread_id, bounds, num_threads):
    """Запустить оптимизацию в отдельном потоке"""
    print(f"[Thread {thread_id}] Запуск оптимизации...")
    
    optimizer = EMMOEA(
        num_pop=100,
        num_obj=2,
        num_var=10,
        bounds=bounds,
        problem=UF7,
        surrogate='MultiTaskIBNN',
        max_evals=4500 // num_threads,
        gmax=10
    )
    
    decisions, objectives = optimizer.optimize()
    print(f"[Thread {thread_id}] Завершено! {len(decisions)} решений")
    
    return decisions, objectives


def run_example():
    bounds = np.array([
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
    ])
    
    num_threads = 30
    
    print("="*70)
    print(f"EMMOEA - Оптимизация UF7 на {num_threads} потоках")
    print("="*70)
    
    # Запускаем оптимизацию в нескольких потоках
    all_decisions = []
    all_objectives = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(run_optimization_thread, i, bounds, num_threads)
            for i in range(num_threads)
        ]
        
        for future in futures:
            decisions, objectives = future.result()
            all_decisions.append(decisions)
            all_objectives.append(objectives)
    
    # Объединяем результаты со всех потоков
    combined_decisions = np.vstack(all_decisions)
    combined_objectives = np.vstack(all_objectives)
    
    print(f"\nВсего получено решений: {len(combined_objectives)}")
    
    # Получаем только недоминируемые решения
    nds = NonDominatedSorting()
    front_indices = nds.do(combined_objectives)[0]
    pareto_decisions = combined_decisions[front_indices]
    pareto_objectives = combined_objectives[front_indices]
    
    print(f"Решений на фронте Парето: {len(pareto_objectives)}")
    
    # Генерируем истинный фронт Парето для UF7
    x_true = np.linspace(0, 1, 1000).reshape(-1, 1)
    y_true = 1 - x_true
    true_front = np.hstack([x_true, y_true])
    
    # Вычисляем IGD для каждого потока
    igd_values = []
    print("\n" + "="*70)
    print("IGD (Inverted Generational Distance) по потокам:")
    print("="*70)
    for i in range(num_threads):
        igd = compute_igd(all_objectives[i], true_front)
        igd_values.append(igd)
        print(f"Thread {i}: IGD = {igd:.6f}")
    
    # Среднее значение IGD
    mean_igd = np.mean(igd_values)
    std_igd = np.std(igd_values)
    print("="*70)
    print(f"Среднее IGD:    {mean_igd:.6f}")
    print(f"Стд. отклон.:   {std_igd:.6f}")
    
    # IGD для итогового фронта Парето
    pareto_igd = compute_igd(pareto_objectives, true_front)
    print(f"IGD (Парето):   {pareto_igd:.6f}")
    print("="*70)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    
    # График 1: Все решения со всех потоков
    plt.subplot(1, 2, 1)
    for i in range(num_threads):
        plt.scatter(
            all_objectives[i][:, 0], 
            all_objectives[i][:, 1], 
            alpha=0.5, 
            s=20, 
            label=f'Thread {i} (IGD={igd_values[i]:.4f})'
        )
    
    x_true_plot = np.linspace(0, 1, 1000)
    y_true_plot = 1 - x_true_plot
    plt.plot(x_true_plot, y_true_plot, 'r--', label='Истинный фронт', linewidth=2)
    
    plt.xlabel('f1', fontsize=11)
    plt.ylabel('f2', fontsize=11)
    plt.title('Результаты по потокам', fontsize=12)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # График 2: Фронт Парето
    plt.subplot(1, 2, 2)
    plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], 
               alpha=0.8, s=40, color='blue', label=f'Парето фронт (IGD={pareto_igd:.4f})')
    plt.plot(x_true_plot, y_true_plot, 'r--', label='Истинный фронт', linewidth=2)
    
    plt.xlabel('f1', fontsize=11)
    plt.ylabel('f2', fontsize=11)
    plt.title(f'Фронт Парето ({len(pareto_objectives)} решений)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('emmoea_threads_result.png', dpi=150)
    print("\n✓ График сохранён как 'emmoea_threads_result.png'")
    plt.show()


if __name__ == "__main__":
    run_example()
