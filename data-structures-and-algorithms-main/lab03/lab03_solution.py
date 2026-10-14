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
import random
import statistics
import time

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
    
    # Bubble sort (стабильная)
    bubble_res, _, _ = bubble_sort(data)
    print(f"\n1. Bubble Sort (пузырьком):")
    print(f"   Результат: {bubble_res}")
    print(f"   Метки при ключе 1: {[p[1] for p in bubble_res if p[0] == 1]}")
    print(f"   Метки при ключе 2: {[p[1] for p in bubble_res if p[0] == 2]}")
    print(f"   → СТАБИЛЬНА (порядок меток сохранён: b,d и a,c,f)")
    
    # Insertion sort (стабильная)
    insertion_res, _, _ = insertion_sort(data)
    print(f"\n2. Insertion Sort (вставками):")
    print(f"   Результат: {insertion_res}")
    print(f"   Метки при ключе 1: {[p[1] for p in insertion_res if p[0] == 1]}")
    print(f"   Метки при ключе 2: {[p[1] for p in insertion_res if p[0] == 2]}")
    print(f"   → СТАБИЛЬНА (порядок меток сохранён)")
    
    # Selection sort (нестабильная)
    selection_res, _, _ = selection_sort(data)
    print(f"\n3. Selection Sort (выбором):")
    print(f"   Результат: {selection_res}")
    print(f"   Метки при ключе 1: {[p[1] for p in selection_res if p[0] == 1]}")
    print(f"   Метки при ключе 2: {[p[1] for p in selection_res if p[0] == 2]}")
    print(f"   → НЕСТАБИЛЬНА (порядок меток может нарушаться)")
    
    # QuickSort (нестабильная)
    quick_res = quick_sort(data, random.Random(42))
    print(f"\n4. Quick Sort (быстрая):")
    print(f"   Результат: {quick_res}")
    print(f"   Метки при ключе 1: {[p[1] for p in quick_res if p[0] == 1]}")
    print(f"   Метки при ключе 2: {[p[1] for p in quick_res if p[0] == 2]}")
    print(f"   → НЕСТАБИЛЬНА (порядок меток может нарушаться)")
    
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


def run_benchmarks(seed: int) -> None:
    """Замеры всех алгоритмов на трёх классах входов; результат — таблица в stdout."""
    plans = [
        ("bubble_sort", ALGORITHMS["bubble_sort"], SIZES_QUADRATIC),
        ("insertion_sort", ALGORITHMS["insertion_sort"], SIZES_QUADRATIC),
        ("selection_sort", ALGORITHMS["selection_sort"], SIZES_QUADRATIC),
        ("quick_sort", lambda a: quick_sort(a, random.Random(seed)), SIZES_QUICK),
    ]
    for name, fn, sizes in plans:
        print(f"\n{name}:")
        for n in sizes:
            for cls, data in make_inputs(n, seed).items():
                print(f"  n={n:>7}  вход={cls:<13} t={bench(fn, data):.6f} c")
    # TODO: построить log-log графики (matplotlib) по классам входов;
    # TODO: сопоставить наклоны с O(n^2) и O(n log n), объяснить расхождения
    #       (в т. ч. лучший случай вставок и поведение пузырька на упорядоченном входе);
    # TODO: включить в отчёт таблицы счётчиков сравнений/обменов.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = ap.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()
    stability_demo()
    run_benchmarks(seed)


if __name__ == "__main__":
    main()
