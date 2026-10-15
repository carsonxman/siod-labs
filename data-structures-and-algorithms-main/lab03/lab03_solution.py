#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 3. Простые сортировки и QuickSort с рандомизацией — стартовая заготовка.

Заготовка — это каркас, а не решение: заполните все TODO самостоятельно
в соответствии с КИМ-03 и правилами использования генеративного ИИ
(docs/ai-verification.md).

Запуск: python lab03-simple-sorts-quicksort-starter.py --variant N
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Ошибка: matplotlib не найден.")
# ---------------------------------------------------------------------------
# 1. Простые сортировки со счётчиками (сортируют КОПИЮ входа, вход не меняют)
#    Каждая функция возвращает кортеж: (отсортированный список,
#    число сравнений ключей, число обменов/перемещений элементов).
# ---------------------------------------------------------------------------


def bubble_sort(a: list) -> tuple[list, int, int]:
    """Сортировка пузырьком со счётчиками.
    
    Сложность: O(n²) в среднем и худшем случае, O(n) в лучшем (упорядоченный вход).
    Стабильная сортировка.
    """
    arr = a.copy()
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
                swapped = True
        if not swapped:  # досрочный выход — массив уже отсортирован
            break
    
    return arr, comparisons, swaps


def insertion_sort(a: list) -> tuple[list, int, int]:
    """Сортировка вставками со счётчиками.
    
    Сложность: O(n²) в среднем и худшем случае, O(n) в лучшем (упорядоченный вход).
    Стабильная сортировка.
    """
    arr = a.copy()
    n = len(arr)
    comparisons = 0
    shifts = 0  # перемещения (сдвиги элементов)
    
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        # Сдвигаем элементы, большие key, на одну позицию вправо
        while j >= 0:
            comparisons += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                shifts += 1
                j -= 1
            else:
                break
        arr[j + 1] = key
    
    return arr, comparisons, shifts


def selection_sort(a: list) -> tuple[list, int, int]:
    """Сортировка выбором со счётчиками.
    
    Сложность: O(n²) сравнений на ЛЮБОМ входе (всегда n*(n-1)/2 сравнений).
    Нестабильная сортировка.
    """
    arr = a.copy()
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            comparisons += 1
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            swaps += 1
    
    return arr, comparisons, swaps


# ---------------------------------------------------------------------------
# 2. QuickSort с рандомизированным опорным элементом
# ---------------------------------------------------------------------------


def quick_sort(a: list, rng: random.Random | None = None) -> list:
    """QuickSort с рандомизированным выбором опорного элемента.
    
    Сложность: O(n log n) в среднем, O(n²) в худшем случае.
    Нестабильная сортировка.
    
    Args:
        a: входной список
        rng: генератор случайных чисел (для воспроизводимости)
    
    Returns:
        новый отсортированный список
    """
    if rng is None:
        rng = random.Random()
    
    arr = a.copy()
    
    def _quick_sort(lo: int, hi: int) -> None:
        if lo >= hi:
            return
        
        # Рандомизированный выбор опорного элемента
        pivot_idx = rng.randrange(lo, hi + 1)
        pivot = arr[pivot_idx]
        
        # Разбиение Ломуто
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]  # перемещаем пивот в конец
        store_idx = lo
        
        for i in range(lo, hi):
            if arr[i] <= pivot:
                arr[store_idx], arr[i] = arr[i], arr[store_idx]
                store_idx += 1
        
        arr[store_idx], arr[hi] = arr[hi], arr[store_idx]  # пивот на своё место
        
        _quick_sort(lo, store_idx - 1)
        _quick_sort(store_idx + 1, hi)
    
    _quick_sort(0, len(arr) - 1)
    return arr


# ---------------------------------------------------------------------------
# 3. Демонстрация стабильности на парах (ключ, метка)
# ---------------------------------------------------------------------------


def stability_demo() -> None:
    """Демонстрация стабильности/нестабильности сортировок."""
    print("\n=== ДЕМОНСТРАЦИЯ СТАБИЛЬНОСТИ ===")
    
    # Массив пар (ключ, метка) с повторяющимися ключами
    data = [(2, "a"), (1, "b"), (2, "c"), (1, "d"), (3, "e"), (2, "f")]
    print(f"Исходный массив: {data}")
    
    # Вспомогательная функция для проверки стабильности
    def check_stability(result_list: list[tuple[int, str]], original_list: list[tuple[int, str]]) -> bool:
        # Группируем метки по ключам в исходном списке
        orig_groups = {}
        for k, v in original_list:
            orig_groups.setdefault(k, []).append(v)
        
        # Группируем метки по ключам в отсортированном списке
        res_groups = {}
        for k, v in result_list:
            res_groups.setdefault(k, []).append(v)
            
        # Сравниваем порядок внутри групп
        for k in orig_groups:
            if orig_groups[k] != res_groups.get(k, []):
                return False
        return True

    algorithms = [
        ("Bubble Sort", bubble_sort(data)[0]),
        ("Insertion Sort", insertion_sort(data)[0]),
        ("Selection Sort", selection_sort(data)[0]),
        ("QuickSort", quick_sort(data, random.Random(42))),
    ]

    for name, res in algorithms:
        is_stable = check_stability(res, data)
        status = "СТАБИЛЬНА" if is_stable else "НЕСТАБИЛЬНА"
        
        print(f"\n{name}: {status}")
        print(f"   Результат: {res}")
        
        # Выводим детали только если интересно (можно убрать, чтобы не спамить консоль)
        keys_in_res = sorted(set(k for k, _ in res))
        for k in keys_in_res:
            labels_orig = [v for kk, v in data if kk == k]
            labels_res = [v for kk, v in res if kk == k]
            match_str = "✓" if labels_orig == labels_res else "✗ НАРУШЕН"
            print(f"   Ключ {k}: было {labels_orig} -> стало {labels_res} [{match_str}]")

    print("\n=== ВЫВОДЫ О СТАБИЛЬНОСТИ ===")
    print("Стабильные: Bubble Sort, Insertion Sort")
    print("Нестабильные: Selection Sort, QuickSort")
    print("\nСтабильность важна в задачах, где требуется сортировка по нескольким")
    print("ключам: сначала по одному, потом по другому с сохранением порядка.")


# ---------------------------------------------------------------------------
# 4. Классы входов (данные по варианту, детерминированно по seed)
# ---------------------------------------------------------------------------


def make_inputs(n: int, seed: int) -> dict[str, list[int]]:
    """Три класса входов размера n: упорядоченный, случайный, обратный."""
    rng = random.Random(seed + n)  # свой поток для каждого размера
    data = [rng.randint(-1_000_000, 1_000_000) for _ in range(n)]
    return {
        "упорядоченный": sorted(data),
        "случайный": data,
        "обратный": sorted(data, reverse=True),
    }


# ---------------------------------------------------------------------------
# 5. Верификация (шаги 1–3 методики docs/ai-verification.md)
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "bubble_sort": lambda a: bubble_sort(a)[0],
    "insertion_sort": lambda a: insertion_sort(a)[0],
    "selection_sort": lambda a: selection_sort(a)[0],
    "quick_sort": lambda a: quick_sort(a, random.Random(0)),
}


def is_sorted(a: list) -> bool:
    """Инвариант 1: неубывающий порядок элементов."""
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


def self_check() -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],                    # пустой массив
        [7],                   # один элемент
        [5, 5, 5, 5],          # все элементы равны
        [1, 2, 3, 4, 5],       # уже отсортирован
        [5, 4, 3, 2, 1],       # обратный порядок
    ]
    for name, fn in ALGORITHMS.items():
        for a in boundary:
            res = fn(list(a))
            assert is_sorted(res), f"{name}: нарушен порядок на {a}"
            assert sorted(res) == sorted(a), f"{name}: не перестановка входа {a}"

    # Сверка с эталоном из стандартной библиотеки на сотнях случайных входов.
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.randint(-100, 100) for _ in range(rng.randint(0, 80))]
        expected = sorted(a)
        for name, fn in ALGORITHMS.items():
            assert fn(list(a)) == expected, f"{name}: расходится с sorted() на {a}"

    # Счётчики: у сортировки выбором ровно n*(n-1)/2 сравнений на любом входе.
    _, comparisons, _ = selection_sort([3, 1, 2, 5, 4])
    assert comparisons == 10, "selection_sort: неверный счётчик сравнений"

        # --- Инвариант 1: число обменов bubble_sort = число инверсий ---
    def count_inversions(a: list) -> int:
        """Число пар (i, j), i < j, a[i] > a[j]."""
        return sum(
            1
            for i in range(len(a))
            for j in range(i + 1, len(a))
            if a[i] > a[j]
        )

    rng = random.Random(0)
    for _ in range(50):
        a = [rng.randint(-50, 50) for _ in range(rng.randint(0, 40))]
        _, _, swaps = bubble_sort(a)
        inv = count_inversions(a)
        assert swaps == inv, (
            f"bubble_sort: обменов {swaps}, инверсий {inv} на {a}"
        )
    print("  [1] bubble_sort: обмены == инверсии — OK")

    # --- Инвариант 2: досрочный выход на упорядоченном входе ---
    for n in [1, 2, 5, 10, 100]:
        sorted_input = list(range(n))
        _, comparisons, _ = bubble_sort(sorted_input)
        expected = max(n - 1, 0)
        assert comparisons == expected, (
            f"bubble_sort: на упорядоченном n={n} сравнений "
            f"{comparisons}, ожидалось {expected}"
        )
    print("  [2] bubble_sort: n-1 сравнений на упорядоченном входе — OK")

    # --- Инвариант 3: число сдвигов insertion_sort = число инверсий ---
    for _ in range(50):
        a = [rng.randint(-50, 50) for _ in range(rng.randint(0, 40))]
        _, _, shifts = insertion_sort(a)
        inv = count_inversions(a)
        assert shifts == inv, (
            f"insertion_sort: сдвигов {shifts}, инверсий {inv} на {a}"
        )
    print("  [3] insertion_sort: сдвиги == инверсии — OK")

    # --- Инвариант 4: selection_sort всегда n(n-1)/2 сравнений ---
    for n in [1, 2, 5, 10, 100]:
        expected = n * (n - 1) // 2
        for cls_name, data in make_inputs(n, seed=42).items():
            _, comparisons, _ = selection_sort(data)
            assert comparisons == expected, (
                f"selection_sort: на n={n} ({cls_name}) сравнений "
                f"{comparisons}, ожидалось {expected}"
            )
    print("  [4] selection_sort: n(n-1)/2 сравнений на любом входе — OK")

    print("self_check: OK")



# ---------------------------------------------------------------------------
# 6. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES_QUADRATIC = [500, 1_000, 2_000, 4_000, 8_000]     # для простых сортировок
SIZES_QUICK = [1_000, 3_000, 10_000, 30_000, 100_000]   # для QuickSort
REPEATS = 5


def bench(fn, data: list) -> float:
    """Медиана времени выполнения fn(копия data) по REPEATS запускам, с прогревом."""
    fn(list(data))  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        arg = list(data)  # свежая копия: сортировка не должна получать свой результат
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def estimate_slope(points: list[tuple[int, float]]) -> float:
    if len(points) < 2:
        return float("nan")

    xs = []
    ys = []

    for n, t in points:
        if n <= 0:
            continue
        xs.append(math.log10(n))
        ys.append(math.log10(max(t, 1e-12)))

    if len(xs) < 2:
        return float("nan")

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) * (x - x_mean) for x in xs)

    if denominator == 0.0:
        return float("nan")

    return numerator / denominator


def plot_results(results: dict, variant: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    algorithms = ["bubble_sort", "insertion_sort", "selection_sort", "quick_sort"]
    titles = {
        "bubble_sort": "Bubble Sort",
        "insertion_sort": "Insertion Sort",
        "selection_sort": "Selection Sort",
        "quick_sort": "QuickSort",
    }
    colors = {
        "упорядоченный": "green",
        "случайный": "blue",
        "обратный": "red",
    }

    for idx, alg in enumerate(algorithms):
        ax = axes[idx]
        data = results[alg]

        for cls in ["упорядоченный", "случайный", "обратный"]:
            points = data[cls]
            sizes = [p[0] for p in points]
            times = [max(p[1], 1e-12) for p in points]
            ax.loglog(
                sizes,
                times,
                "o-",
                label=cls,
                color=colors[cls],
                linewidth=2,
                markersize=6,
            )

        ref_points = data["случайный"]
        if ref_points:
            first_n, raw_first_t = ref_points[0]
            first_t = max(raw_first_t, 1e-12)
            n_ref = [p[0] for p in ref_points]

            if alg == "quick_sort":
                if first_n > 1:
                    base = first_n * math.log2(first_n)
                    if base > 0:
                        t_ref = []
                        for n in n_ref:
                            if n > 1:
                                t_ref.append(first_t * (n * math.log2(n) / base))
                            else:
                                t_ref.append(first_t)

                        ax.loglog(
                            n_ref,
                            t_ref,
                            "--",
                            color="gray",
                            alpha=0.6,
                            label="ориентир O(n log n)",
                        )
            else:
                if first_n > 0:
                    t_ref = [first_t * ((n / first_n) ** 2) for n in n_ref]
                    ax.loglog(
                        n_ref,
                        t_ref,
                        "--",
                        color="gray",
                        alpha=0.6,
                        label="ориентир O(n^2)",
                    )

        ax.set_xlabel("Размер входа n")
        ax.set_ylabel("Время, с")
        ax.set_title(titles[alg])
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"ЛР 3. Сравнение сортировок, вариант {variant}", fontsize=15)
    plt.tight_layout()

    out_path = Path(__file__).resolve().with_name(f"lab03_graphs_variant{variant}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"График сохранён: {out_path.name}")


def print_counters_table(seed: int) -> None:
    sizes = [100, 500, 1000]

    print("\n=== ТАБЛИЦЫ СЧЁТЧИКОВ ===")

    for alg_name, alg_func in [
        ("Bubble Sort", bubble_sort),
        ("Insertion Sort", insertion_sort),
        ("Selection Sort", selection_sort),
    ]:
        print(f"\n{alg_name}:")
        print(f"{'n':>6} | {'Класс':<13} | {'Сравнения':>10} | {'Обмены/Сдвиги':>14}")
        print("-" * 58)

        for n in sizes:
            inputs = make_inputs(n, seed)
            for cls, data in inputs.items():
                _, comp, moves = alg_func(data)
                print(f"{n:>6} | {cls:<13} | {comp:>10} | {moves:>14}")


def run_benchmarks(seed: int, variant: int) -> None:
    print("\n=== БЕНЧМАРКИ ===")
    print(f"Вариант: {variant}, Seed: {seed}")

    plans = [
        ("bubble_sort", ALGORITHMS["bubble_sort"], SIZES_QUADRATIC),
        ("insertion_sort", ALGORITHMS["insertion_sort"], SIZES_QUADRATIC),
        ("selection_sort", ALGORITHMS["selection_sort"], SIZES_QUADRATIC),
        ("quick_sort", lambda a: quick_sort(a, random.Random(seed)), SIZES_QUICK),
    ]

    # Структура: {alg_name: {class_name: [(size, time), ...]}}
    results = {}

    for name, fn, sizes in plans:
        print(f"\n{name}:")
        results[name] = {"упорядоченный": [], "случайный": [], "обратный": []}
        
        for n in sizes:
            inputs = make_inputs(n, seed)
            for cls, data in inputs.items():
                t = bench(fn, data)
                results[name][cls].append((n, t))
                print(f"  n={n:>7}  вход={cls:<13} t={t:.6f} c")

    # Вызываем построение графиков ПЕРЕД выходом из функции
    plot_results(results, variant)
    
    # Опционально: таблицы счетчиков
    print_counters_table(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="ЛР 3")
    ap.add_argument("--variant", type=int, required=True)
    args = ap.parse_args()

    seed = 30 + args.variant
    random.seed(seed)

    self_check()
    stability_demo()
    run_benchmarks(seed, args.variant)


if __name__ == "__main__":
    main()
