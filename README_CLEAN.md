# EMMOEA - Efficient Multi-Objective Surrogate-Assisted Evolutionary Algorithm

Реализация многоцелевого эволюционного алгоритма с суррогатными моделями.

## Модули

### `emmoea/optimizer.py`
Основной класс **EMMOEA** с методами:
- `SimplexLattice()` - генерирование точек на решётке
- `NBI()` - Normal Boundary Intersection
- `Latin()` - Latin Hypercube Sampling
- `UniformPoint()` - выбор метода сэмплирования
- `scale_and_evaluate()` - масштабирование и оценка
- `GAreal()` - генетические операторы (SBX + мутация)
- `optimize()` - главный метод оптимизации

### `emmoea/utils/sampling.py`
Методы генерирования равномерных точек (вспомогательные функции):
- `SimplexLattice(H, M)` - симплексная решётка
- `NBI(N, M)` - normal boundary intersection
- `Latin(N, M)` - latin hypercube
- `UniformPoint(N, M, method)` - обёртка для выбора метода

### `emmoea/utils/ga_operators.py`
Генетические операторы:
- `GAreal(Population, bounds, ...)` - SBX кроссовер + полиномиальная мутация

### `emmoea/utils/selection.py`
Методы отбора решений:
- `kriging_selection(PopObj, V)` - отбор на основе расстояний до reference vectors

### `emmoea/problems/__init__.py`
Тестовые функции:
- `UF7(X)` - 2 объектива, 10 переменных
- `UF1(X)` - 2 объектива, 30 переменных
- `ZDT1(X)` - классическая функция, 30 переменных

## Установка

```bash
# Создать виртуальное окружение
python -m venv venv
.\venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt

# Установить пакет в режиме разработки (опционально)
pip install -e .
```

## Использование

```python
from emmoea import EMMOEA
from emmoea.problems import UF7
import numpy as np

# Определить границы
bounds = np.array([[0, -1, -1, ...], [1, 1, 1, ...]])

# Создать оптимизатор
optimizer = EMMOEA(
    num_pop=100,      # Размер популяции
    num_obj=2,        # Количество объективов
    num_var=10,       # Количество переменных
    bounds=bounds,
    problem=UF7,
    surrogate='KRG',  # или 'DACE'
    max_evals=500,
    gmax=10
)

# Запустить оптимизацию
decisions, objectives = optimizer.optimize()
```

## Основные компоненты

### Сэмплирование (Sampling)
- **NBI** - Normal Boundary Intersection для равномерного распределения точек
- **Latin** - Latin Hypercube для псевдослучайного сэмплирования

### Генетические операторы
- **SBX** (Simulated Binary Crossover) - кроссовер
- **Полиномиальная мутация** - для мутации

### Суррогатные модели
- **KRG** (Kriging) - модель из SMT Toolbox
- **DACE** - Design and Analysis of Computer Experiments из pydacefit

### Отбор решений
- **kriging_selection()** - отбор на основе расстояний до reference vectors

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
python tests/
```

## Авторы

Сергей

## Лицензия

MIT
