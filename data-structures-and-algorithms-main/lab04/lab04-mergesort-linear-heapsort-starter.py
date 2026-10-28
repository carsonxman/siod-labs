#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 4. MergeSort, Counting, Radix и HeapSort: оценка производительности.

Студент: Данилов Всеволод Александрович, группа БВТ2503
Вариант: 8 (seed = 30 + 8 = 38)

Запуск: python lab04-mergesort-linear-heapsort-starter.py --variant 8
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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

    Сложность: O(n log n) в лучшем, среднем и худшем случаях.
    Память: O(n) вспомогательной памяти под слияние.
    Стабильная сортировка.
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
    Сложность по времени: O(n + k), где n = len(a), k = key_max + 1.
    Сложность по памяти: O(n + k) (массив результата и массив счётчиков).
    Устойчивость гарантируется обратным проходом при расстановке.
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
    Сложность: O(d * (n + b)), где d — число разрядов, b = 10 — основание.
    Память: O(n + b).
    Устойчивость внутреннего прохода обязательна для сохранения порядка
    чисел по уже отсортированным младшим разрядам.
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

    Схема: построение max-кучи просейками вниз снизу вверх (O(n)),
    затем обмен вершины с последним элементом и просейка (n-1 раз по O(log n)).
    Сложность: O(n log n) во всех случаях. Память: O(1).
    Нестабильная сортировка.
    """
    n = len(a)
    if n <= 1:
        return

    # 1. Построение max-кучи снизу вверх за O(n)
    for i in range(n // 2 - 1, -1, -1):
        sift_down(a, i, n)

    # 2. Фаза извлечения максимумов
    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        sift_down(a, 0, i)


def quick_sort(a: list[int], rng: random.Random | None = None) -> list[int]:
    """QuickSort с рандомизацией опорного элемента — перенесено из ЛР 3 (КИМ-03).

    Разбиение Ломуто с хвостовой оптимизацией глубины стека O(log n).
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
#    (шаги 1–3 методики docs/ai-verification.md)
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
    """Собственная проверка: проверка свойства max-кучи после фазы построения."""
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
                expect(a[i] >= a[left], f"Нарушено свойство кучи: a[{i}] < a[{left}]")
            if right < n:
                expect(a[i] >= a[right], f"Нарушено свойство кучи: a[{i}] < a[{right}]")


def check_radix_stability_requirement() -> None:
    """Собственная проверка: подтверждение обязательности устойчивости поразрядного прохода."""
    data = [21, 11]
    res = radix_sort_lsd(data)
    expect(res == [11, 21], f"radix_sort_lsd ошибся на {data}: получено {res}")


def self_check() -> None:
    """Все проверки по разделам; при любой ошибке замеры не выполняются."""
    checks = [(name, lambda fn=fn: check_sort(fn)) for name, fn in SORTS.items()]
    checks.append(("стабильность merge_sort", check_merge_stability))
    checks.append(("свойство max-кучи (heapify)", check_heap_property))
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


# ---------------------------------------------------------------------------
# 3. Бенчмарк (методика — docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
SMALL_KEY_MAX = 100        # узкий диапазон: counting уместен
WIDE_KEY_MAX = 10**9       # широкий диапазон: counting теряет смысл


def make_datasets(rng: random.Random, n: int) -> dict[str, list[int]]:
    """Массивы разных распределений и диапазонов ключей для размера n."""
    almost = list(range(n))
    for _ in range(max(1, n // 100)):  # ~1 % случайных обменов
        i, j = rng.randrange(n), rng.randrange(n)
        almost[i], almost[j] = almost[j], almost[i]
    return {
        "random_wide": [rng.randint(0, WIDE_KEY_MAX) for _ in range(n)],
        "random_small": [rng.randint(0, SMALL_KEY_MAX) for _ in range(n)],
        "sorted": list(range(n)),
        "reversed": list(range(n, 0, -1)),
        "almost_sorted": almost,
    }


def bench(fn, data: list[int]) -> float:
    """Медиана времени fn на копии data по REPEATS запускам, с прогревом."""
    fn(list(data))  # прогрев — не учитывается
    times = []
    for _ in range(REPEATS):
        buf = list(data)
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def compute_slope(points: list[tuple[int, float]]) -> float:
    """Вычисляет наклон прямой в log-log координатах: log10(t) = k * log10(n) + b."""
    valid_points = [(n, t) for n, t in points if t > 0]
    if len(valid_points) < 2:
        return 0.0
    log_x = [math.log10(n) for n, _ in valid_points]
    log_y = [math.log10(t) for _, t in valid_points]
    mean_x = statistics.mean(log_x)
    mean_y = statistics.mean(log_y)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y))
    den = sum((x - mean_x) ** 2 for x in log_x)
    return num / den if den != 0 else 0.0


def plot_results(results: dict[str, dict[str, list[tuple[int, float]]]], variant: int) -> None:
    """Построение log-log графиков (matplotlib) по каждому распределению."""
    if plt is None:
        print("Предупреждение: matplotlib не установлен, график не построен.")
        return

    distributions = ["random_wide", "random_small", "sorted", "reversed", "almost_sorted"]
    dist_titles = {
        "random_wide": "Случайный широкий (keys in [0, 10^9])",
        "random_small": "Случайный узкий (keys in [0, 100])",
        "sorted": "Отсортированный (sorted)",
        "reversed": "Обратно отсортированный (reversed)",
        "almost_sorted": "Почти отсортированный (~1% обменов)",
    }

    colors = {
        "merge_sort": "#1f77b4",
        "counting_sort": "#ff7f0e",
        "radix_sort_lsd": "#2ca02c",
        "heap_sort": "#d62728",
        "quick_sort": "#9467bd",
        "sorted": "#8c564b",
    }

    markers = {
        "merge_sort": "o",
        "counting_sort": "s",
        "radix_sort_lsd": "^",
        "heap_sort": "D",
        "quick_sort": "v",
        "sorted": "*",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, dist in enumerate(distributions):
        ax = axes_flat[idx]
        for algo_name, dist_data in results.items():
            pts = dist_data[dist]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs,
                ys,
                label=algo_name,
                color=colors.get(algo_name, "black"),
                marker=markers.get(algo_name, "o"),
                linewidth=1.8,
                markersize=6,
            )

        ref_x = SIZES
        first_quick = results["quick_sort"][dist][0][1] if results["quick_sort"][dist] else 1e-4
        c_nlogn = first_quick / (ref_x[0] * math.log2(ref_x[0]))
        ref_y_nlogn = [c_nlogn * x * math.log2(x) for x in ref_x]
        ax.plot(ref_x, ref_y_nlogn, "--", color="gray", alpha=0.5, label="ориентир O(n log n)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Размер массива n", fontsize=10)
        ax.set_ylabel("Время, с", fontsize=10)
        ax.set_title(dist_titles[dist], fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    axes_flat[5].axis("off")
    summary_text = (
        f"ЛР 4. Вариант {variant} (seed = {30 + variant})\n"
        f"Сравнение 6 алгоритмов сортировки:\n\n"
        f"- MergeSort: гарантированный O(n log n), стабильный\n"
        f"- CountingSort: O(n + k), стабильный, быстрый при малом k\n"
        f"- RadixSort LSD: O(d * n), стабильный поразрядный\n"
        f"- HeapSort: in-place O(n log n), нестабильный\n"
        f"- QuickSort: рандомизированный O(n log n), из ЛР 3\n"
        f"- sorted(): встроенный эталон Python (Timsort)"
    )
    axes_flat[5].text(
        0.1, 0.5, summary_text,
        fontsize=11, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=1", facecolor="#f0f0f0", edgecolor="#ccc")
    )

    fig.suptitle(f"ЛР 4. Экспериментальное сравнение сортировок (Вариант {variant})", fontsize=16, fontweight="bold")
    plt.tight_layout()

    out_path = Path(__file__).resolve().with_name(f"lab04_graphs_variant{variant}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик сохранён: {out_path.name}")


def run_benchmarks(seed: int, variant: int) -> None:
    print(f"=== БЕНЧМАРКИ (Вариант {variant}, Seed = {seed}) ===")
    rng = random.Random(seed)

    algorithms = [
        ("merge_sort", merge_sort),
        ("counting_sort", lambda a: counting_sort(a, key_max=max(a, default=0))),
        ("radix_sort_lsd", radix_sort_lsd),
        ("heap_sort", heap_sort),
        ("quick_sort", lambda a: quick_sort(a, random.Random(seed))),
        ("sorted", sorted),
    ]

    distributions = ["random_wide", "random_small", "sorted", "reversed", "almost_sorted"]

    results: dict[str, dict[str, list[tuple[int, float]]]] = {
        algo[0]: {dist: [] for dist in distributions} for algo in algorithms
    }

    all_datasets = {n: make_datasets(rng, n) for n in SIZES}

    for n in SIZES:
        print(f"\n--- n = {n} ---")
        datasets = all_datasets[n]
        for dist_name in distributions:
            data = datasets[dist_name]
            print(f"  [{dist_name}]:")
            for algo_name, fn in algorithms:
                if algo_name == "counting_sort" and dist_name == "random_wide":
                    print(f"    {algo_name:>15}: пропуск (k = 10^9 >> n)")
                    continue
                t = bench(fn, data)
                results[algo_name][dist_name].append((n, t))
                print(f"    {algo_name:>15}: t = {t:.6f} c")

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ ТАБЛИЦЫ ДЛЯ ОТЧЕТА (ВРЕМЯ В СЕКУНДАХ):")
    print("=" * 60)

    for dist_name in distributions:
        print(f"\n### Распределение: {dist_name}")
        header = f"| {'N':>7} | " + " | ".join(f"{algo[0]:>15}" for algo in algorithms) + " |"
        sep = f"|{'-' * 9}|" + "|".join(f"{'-' * 17}" for _ in algorithms) + "|"
        print(header)
        print(sep)
        for i, n in enumerate(SIZES):
            row = [f"{n:>7}"]
            for algo_name, _ in algorithms:
                pts = results[algo_name][dist_name]
                matched = [p[1] for p in pts if p[0] == n]
                if matched:
                    row.append(f"{matched[0]:>15.6f}")
                else:
                    row.append(f"{'пропуск':>15}")
            print("| " + " | ".join(row) + " |")

    print("\n" + "=" * 60)
    print("НАКЛОНЫ LOG-LOG РЕГРЕССИИ (k в T(n) ~ C * n^k):")
    print("=" * 60)
    header = f"| {'Алгоритм':<15} | " + " | ".join(f"{d:<13}" for d in distributions) + " |"
    sep = f"|{'-' * 17}|" + "|".join(f"{'-' * 15}" for _ in distributions) + "|"
    print(header)
    print(sep)
    for algo_name, _ in algorithms:
        slopes = []
        for dist in distributions:
            pts = results[algo_name][dist]
            if pts:
                slope = compute_slope(pts)
                slopes.append(f"{slope:>13.2f}")
            else:
                slopes.append(f"{'-':>13}")
        print(f"| {algo_name:<15} | " + " | ".join(slopes) + " |")

    plot_results(results, variant)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта")
    args = ap.parse_args()

    seed = 30 + args.variant
    random.seed(seed)

    self_check()
    run_benchmarks(seed, args.variant)


if __name__ == "__main__":
    main()
