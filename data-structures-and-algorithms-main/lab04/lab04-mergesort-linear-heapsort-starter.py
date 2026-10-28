#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 4. MergeSort, Counting, Radix и HeapSort: оценка производительности — реализация алгоритмов.

Студент: Данилов Всеволод Александрович, группа БВТ2503
Вариант: 8 (seed = 30 + 8 = 38)

Запуск: python lab04-mergesort-linear-heapsort-starter.py --variant N
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Сортировки (реализуются вручную; sorted() используется только как эталон)
# ---------------------------------------------------------------------------


def merge(left: list, right: list, key=lambda x: x) -> list:
    """Стабильное слияние двух отсортированных списков в новый список.

    Стабильность: при равенстве ключей элемент из left идёт раньше (условие <=).
    Сложность: O(len(left) + len(right)) по времени, O(len(left) + len(right)) по памяти.
    """
    res = []
    i = 0
    j = 0
    n_left = len(left)
    n_right = len(right)

    while i < n_left and j < n_right:
        if key(left[i]) <= key(right[j]):
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1

    if i < n_left:
        res.extend(left[i:])
    if j < n_right:
        res.extend(right[j:])

    return res


def merge_sort(a: list, key=lambda x: x) -> list:
    """MergeSort: возвращает новый отсортированный список.

    Сложность: O(n log n) во всех случаях.
    Память: O(n) под слияние.
    """
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    left = merge_sort(a[:mid], key=key)
    right = merge_sort(a[mid:], key=key)
    return merge(left, right, key=key)


def counting_sort(a: list[int], key_max: int) -> list[int]:
    """Устойчивая сортировка подсчётом: ключи — целые из диапазона [0, key_max].

    Схема: массив счётчиков -> префиксные суммы -> расстановка с конца.
    Сложность по времени: O(n + k), где k = key_max + 1.
    Сложность по памяти: O(n + k).
    """
    if not a:
        return []

    counts = [0] * (key_max + 1)
    for x in a:
        counts[x] += 1

    for i in range(1, key_max + 1):
        counts[i] += counts[i - 1]

    res = [0] * len(a)
    for x in reversed(a):
        counts[x] -= 1
        res[counts[x]] = x

    return res


def radix_sort_lsd(a: list[int]) -> list[int]:
    """RadixSort LSD по десятичным разрядам, неотрицательные целые.

    Поверх устойчивой поразрядной counting-сортировки: от младшего разряда
    к старшему.
    Сложность: O(d * (n + b)), b = 10.
    """
    if len(a) <= 1:
        return list(a)

    max_val = max(a)
    if max_val == 0:
        return list(a)

    res = list(a)
    exp = 1
    while max_val // exp > 0:
        counts = [0] * 10
        for x in res:
            digit = (x // exp) % 10
            counts[digit] += 1

        for d in range(1, 10):
            counts[d] += counts[d - 1]

        temp = [0] * len(res)
        for x in reversed(res):
            digit = (x // exp) % 10
            counts[digit] -= 1
            temp[counts[digit]] = x

        res = temp
        exp *= 10

    return res


def sift_down(a: list[int], i: int, n: int) -> None:
    """Просейка вниз элемента i в куче a[0:n] (max-heap), на месте.

    Сложность: O(log n) по времени, O(1) по памяти.
    """
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and a[left] > a[largest]:
            largest = left
        if right < n and a[right] > a[largest]:
            largest = right

        if largest != i:
            a[i], a[largest] = a[largest], a[i]
            i = largest
        else:
            break


def heap_sort(a: list[int]) -> None:
    """HeapSort in-place: сортирует список a на месте, без вспомогательного массива.

    Схема: построение max-кучи за O(n), затем n-1 просеек по O(log n).
    Сложность: O(n log n) во всех случаях, O(1) памяти.
    """
    n = len(a)
    if n <= 1:
        return

    # Фаза 1: построение max-кучи за O(n)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(a, i, n)

    # Фаза 2: сортировка
    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        sift_down(a, 0, i)


def quick_sort(a: list[int], rng: random.Random | None = None) -> list[int]:
    """QuickSort с рандомизацией опорного элемента — перенесено из ЛР 3 (КИМ-03).

    Разбиение Ломуто с оптимизацией хвостовой рекурсии (стек O(log n)).
    """
    if rng is None:
        rng = random.Random()

    arr = list(a)

    def _quick_sort(lo: int, hi: int) -> None:
        while lo < hi:
            pivot_idx = rng.randrange(lo, hi + 1)
            pivot = arr[pivot_idx]
            arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]

            store_idx = lo
            for i in range(lo, hi):
                if arr[i] < pivot:
                    arr[i], arr[store_idx] = arr[store_idx], arr[i]
                    store_idx += 1

            arr[store_idx], arr[hi] = arr[hi], arr[store_idx]

            if store_idx - lo < hi - store_idx:
                _quick_sort(lo, store_idx - 1)
                lo = store_idx + 1
            else:
                _quick_sort(store_idx + 1, hi)
                hi = store_idx - 1

    _quick_sort(0, len(arr) - 1)
    return arr


# ---------------------------------------------------------------------------
# 2. Верификация: инварианты + сверка с эталоном sorted()
# ---------------------------------------------------------------------------


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    """Явная проверка вместо assert: assert-ы исчезают при запуске python -O."""
    if not condition:
        raise SelfCheckError(message)


def heap_sorted(a: list[int]) -> list[int]:
    """heap_sort сортирует на месте: проверяется тот список, который ему передали."""
    buf = list(a)
    heap_sort(buf)
    return buf


SORTS = {
    "merge_sort": lambda a: merge_sort(a),
    "counting_sort": lambda a: counting_sort(a, key_max=999),
    "radix_sort_lsd": lambda a: radix_sort_lsd(a),
    "heap_sort": lambda a: heap_sorted(a),
    "quick_sort": lambda a: quick_sort(a),
}


def check_sort(fn) -> None:
    """Граничные случаи, инварианты сортировки и сверка с эталоном sorted()."""
    boundary = [
        [],                    # пустой массив
        [7],                   # один элемент
        [0],                   # нулевой ключ
        [5, 5, 5],             # все элементы равны
        [1, 2, 3, 4, 5],       # уже отсортирован
        [5, 4, 3, 2, 1],       # обратный порядок
    ]
    for a in boundary:
        res = fn(list(a))
        expect(isinstance(res, list), f"на {a} возвращён {type(res).__name__}, а не список")
        expect(res == sorted(a), f"на {a} получено {res}, ожидалось {sorted(a)}")

    # Случайные тесты: инвариант перестановки + сверка с эталоном
    rng = random.Random(0)
    for _ in range(200):
        a = [rng.randint(0, 999) for _ in range(rng.randint(0, 60))]
        res = fn(list(a))
        expect(sorted(res) == sorted(a), f"не перестановка входа {a}: получено {res}")
        expect(res == sorted(a), f"расходится с sorted() на {a}: получено {res}")


def check_merge_stability() -> None:
    """Стабильность слияния на парах (ключ, порядковый номер): эталон sorted() стабилен,
    поэтому при равных ключах порядок номеров обязан совпасть."""
    rng = random.Random(0)
    pairs = [(rng.randint(0, 5), i) for i in range(100)]
    got = merge_sort(pairs, key=lambda p: p[0])
    expect(
        got == sorted(pairs, key=lambda p: p[0]),
        "merge_sort нестабилен: при равных ключах нарушен исходный порядок номеров",
    )


def check_heap_property() -> None:
    """Собственная проверка: свойство max-кучи после фазы построения O(n)."""
    rng = random.Random(42)
    for _ in range(50):
        n = rng.randint(1, 100)
        a = [rng.randint(0, 1000) for _ in range(n)]
        for i in range(n // 2 - 1, -1, -1):
            sift_down(a, i, n)
        for i in range(n // 2):
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n:
                expect(a[i] >= a[left], f"Нарушено свойство max-кучи: a[{i}] < a[{left}]")
            if right < n:
                expect(a[i] >= a[right], f"Нарушено свойство max-кучи: a[{i}] < a[{right}]")


def check_radix_stability_requirement() -> None:
    """Собственная проверка: подтверждение обязательности устойчивости поразрядного прохода."""
    data = [21, 11]
    res = radix_sort_lsd(data)
    expect(res == [11, 21], f"radix_sort_lsd ошибся на {data}: получено {res}")


def self_check() -> None:
    """Все проверки по разделам; при любой ошибке замеры не выполняются."""
    checks = [(name, lambda fn=fn: check_sort(fn)) for name, fn in SORTS.items()]
    checks.append(("стабильность merge_sort", check_merge_stability))
    checks.append(("свойство кучи (heapify)", check_heap_property))
    checks.append(("устойчивость radix_sort_lsd", check_radix_stability_requirement))

    failed = 0
    print("self_check:")
    for name, check in checks:
        try:
            check()
        except NotImplementedError as err:
            where = traceback.extract_tb(err.__traceback__)[-1].name
            print(f"  {name}: не реализовано — {where} (TODO)")
            failed += 1
        except SelfCheckError as err:
            print(f"  {name}: ОШИБКА — {err}")
            failed += 1
        except Exception:
            print(f"  {name}: ИСКЛЮЧЕНИЕ")
            traceback.print_exc(file=sys.stdout)
            failed += 1
        else:
            print(f"  {name}: OK")

    if failed:
        raise SystemExit(
            f"self_check: не пройдено разделов — {failed}; "
            f"замеры выполняются только после всех проверок."
        )
    print("self_check: OK\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = ap.parse_args()
    seed = 30 + args.variant
    random.seed(seed)
    self_check()


if __name__ == "__main__":
    main()
