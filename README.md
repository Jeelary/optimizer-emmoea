# EMMOEA - Efficient Multi-Objective Surrogate-Assisted Evolutionary Algorithm

Реализация многоцелевого эволюционного алгоритма с суррогатными моделями. Основана на статье "A Performance Indicator-Based Infill Criterion for Expensive Multi-/Many-Objective Optimization" (Shufen Qin, Chaoli Sun, Qiqi Liu, Yaochu Jin).

## Структура проекта

```
optimizer_emmoea/                  # Основной пакет
├── __init__.py
├── optimizer.py                   # Основной класс EMMOEA
├── problems/                      # Тестовые функции
│   ├── __init__.py
│   └── test_functions.py          # Реализация UF*, ZDT* функций
├── utils/                         # Вспомогательные функции
│   ├── __init__.py
│   ├── sampling.py                # Методы сэмплирования (NBI, Latin, Simplex)
│   ├── ga_operators.py            # Генетические операторы (SBX, мутация)
│   └── selection.py               # Методы селекции
├── tests/                         # Модульные тесты
│   └── __init__.py
├── examples/                      # Примеры использования
│   ├── optimize_uf7.py            # Базовый пример оптимизации
│   └── optimize_threads_uf7.py    # Распределённая оптимизация на потоках
├── requirements.txt               # Зависимости проекта
├── setup.py                       # Конфигурация пакета
├── README.md                      # Документация
└── .gitignore                     # Исключения для Git
```

## Установка

```bash
# Создать виртуальное окружение
python -m venv venv
.\venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt

# Установить пакет в режиме разработки
pip install -e .
```

## Быстрый старт

### Пример 1: Базовая оптимизация

```python
from optimizer import EMMOEA
from problems import UF7
import numpy as np

# Определить границы переменных
bounds = np.array([
    [0, -1, -1, -1, -1, -1, -1, -1, -1, -1],  # нижние границы
    [1,  1,  1,  1,  1,  1,  1,  1,  1,  1]   # верхние границы
])

# Создать оптимизатор
optimizer = EMMOEA(
    num_pop=100,           # Размер популяции
    num_obj=2,             # Количество целевых функций
    num_var=10,            # Количество переменных
    bounds=bounds,
    problem=UF7,           # Тестовая функция
    surrogate='DACE',      # Суррогатная модель (KRG или DACE)
    max_evals=5000,        # Максимальное количество оценок
    gmax=10                # Поколений эволюции
)


### Класс EMMOEA

#### Параметры конструктора
```python
EMMOEA(
    num_pop: int,              # Размер популяции
    num_obj: int,              # Количество целевых функций
    num_var: int,              # Количество переменных решения
    bounds: np.ndarray,        # Массив границ переменных [2, num_var]
    problem: callable,         # Функция оценки fitness (принимает X, возвращает F)
    surrogate: str = 'DACE',   # Тип суррогата ('DACE' или 'KRG')
    max_evals: int = 5000,     # Макс. количество оценок функции
    gmax: int = 10             # Количество поколений эволюции
)
```

#### Методы
- `optimize()` - запуск процесса оптимизации
  - **Возвращает:** (decisions, objectives) - решения и их целевые значения

## Основные компоненты

### Модуль `optimizer.py` (EMMOEA)
- **SimplexLattice()** - симплексная решётка для равномерного распределения
- **NBI()** - Normal Boundary Intersection для многоцелевой оптимизации
- **Latin()** - Latin Hypercube Sampling
- **GAreal()** - генетические операторы (SBX кроссовер + полиномиальная мутация)
- **optimize()** - основной метод оптимизации

### Модуль `utils/sampling.py`
- Методы генерирования равномерных точек в целевом пространстве
- UniformPoint() - выбор метода сэмплирования

### Модуль `utils/ga_operators.py`
- **SBX (Simulated Binary Crossover)** - оператор кроссовера
- **Полиномиальная мутация** - оператор мутации

### Модуль `utils/selection.py`
- Методы отбора решений на основе суррогатов
- Использование reference vectors для направления поиска

### Модуль `problems/__init__.py`
Тестовые функции многоцелевой оптимизации:
- **UF7** - 2 целевых функции, 10 переменных
- **UF1** - 2 целевых функции, 30 переменных
- **ZDT1** - классическая функция Zitzler–Deb–Thiele, 30 переменных

## Суррогатные модели

Алгоритм поддерживает две суррогатные модели:
- **DACE** (Design and Analysis of Computer Experiments) - из пакета `pydacefit`, быстрая и надёжная
- **KRG** (Kriging) - из пакета `smt`, более гибкая и точная

## Зависимости

- **numpy** >= 1.20.0 - числовые операции
- **scipy** >= 1.7.0 - научные вычисления
- **scikit-optimize** >= 0.9.0 - базовая оптимизация
- **smt** == 1.3.0 - суррогатные модели (Kriging)
- **pydacefit** >= 0.1.0 - DACE суррогат
- **pymoo** >= 0.5.0 - многоцелевая оптимизация (non-dominated sorting)
- **tqdm** >= 4.60.0 - прогресс-бары
- **matplotlib** >= 3.3.0 - визуализация результатов

## Тестирование

```bash
# Запустить тесты
python -m pytest tests/ -v

# Запустить с покрытием
python -m pytest tests/ --cov=optimizer --cov=utils
```

## Основные источники

1. Qin, S., Sun, C., Liu, Q., & Jin, Y. (2017). A performance indicator-based infill criterion for expensive multi-/many-objective optimization. IEEE Transactions on Evolutionary Computation, 21(3), 370-386.
2. SMT: https://github.com/SMTorg/smt
3. pymoo: https://pymoo.org/

## Основные компоненты

### Оптимизатор (EMMOEA)
- Многоцелевая оптимизация с суррогатными моделями
- Поддержка KRG и DACE суррогатов
- Генетические операторы (SBX кроссовер, полиномиальная мутация)

### Методы сэмплирования
- **NBI** (Normal Boundary Intersection) - равномерное распределение точек
- **Latin** - латинский гиперкуб

### Генетические операторы
- **SBX** (Simulated Binary Crossover)
- **Полиномиальная мутация**

### Суррогатные модели
- **KRG** (Kriging) - от SMT
- **DACE** (Design and Analysis of Computer Experiments) - от pydacefit

## Тестовые функции

- **UF7** - 2 объектива, 10 переменных
- **UF1** - 2 объектива, 30 переменных  
- **ZDT1** - классическая функция, 30 переменных

## Зависимости

- numpy
- scipy
- smt (SMT Toolbox)
- pydacefit
- pymoo
- tqdm

## Примеры

Запуск примера оптимизации:
```bash
python examples/optimize_uf7.py
```

Запуск тестов:
```bash
python -m pytest tests/
```

## Автор

Сергей

## Лицензия

MIT
