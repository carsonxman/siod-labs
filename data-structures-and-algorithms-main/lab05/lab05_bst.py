#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 5. Бинарное дерево поиска: обходы, вставка и удаление.

Студент: Данилов Всеволод Александрович
Группа:  БВТ2503
Вариант: 8 (seed = 38)

Запуск:
    python lab05_bst.py --variant 8
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Бинарное дерево поиска (BST)
# ---------------------------------------------------------------------------


class Node:
    """Узел BST: целочисленный ключ и ссылки на левое и правое поддеревья."""

    __slots__ = ("key", "left", "right")

    def __init__(self, key: int) -> None:
        self.key: int = key
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None


class BST:
    """Бинарное дерево поиска без дубликатов ключей.

    Инвариант BST:
        Для каждого узла x с ключом x.key:
        - для любого узла y в левом поддереве x: y.key < x.key;
        - для любого узла z в правом поддереве x: z.key > x.key.

    Сложность операций:
        - Поиск (search):   O(h)
        - Вставка (insert): O(h)
        - Удаление (delete): O(h)
        где h — высота дерева:
        - В сбалансированном/случайном BST: h = O(log n), операции O(log n);
        - В вырожденном BST (худший случай): h = n - 1, операции O(n).
    """

    def __init__(self) -> None:
        self.root: Optional[Node] = None

    def insert(self, key: int) -> None:
        """Вставка ключа в BST. Повторная вставка существующего ключа игнорируется."""
        if self.root is None:
            self.root = Node(key)
            return

        curr = self.root
        while True:
            if key < curr.key:
                if curr.left is None:
                    curr.left = Node(key)
                    return
                curr = curr.left
            elif key > curr.key:
                if curr.right is None:
                    curr.right = Node(key)
                    return
                curr = curr.right
            else:
                # Ключ уже присутствует — повторная вставка игнорируется
                return

    def search(self, key: int) -> bool:
        """Поиск ключа в BST. Возвращает True, если ключ найден, иначе False."""
        curr = self.root
        while curr is not None:
            if key < curr.key:
                curr = curr.left
            elif key > curr.key:
                curr = curr.right
            else:
                return True
        return False

    def delete(self, key: int) -> None:
        """Удаление ключа из BST с обработкой трёх случаев:

        1) Удаляемый узел — лист (нет детей): отцепляется от родителя;
        2) Удаляемый узел имеет ровно одного ребёнка: узел заменяется этим ребёнком;
        3) Удаляемый узел имеет двух детей: ключ заменяется значением преемника
           (минимальный ключ правого поддерева), после чего преемник рекурсивно
           удаляется из правого поддерева.

        Если ключ отсутствует, дерево не изменяется.
        """
        self.root = self._delete_node(self.root, key)

    def _delete_node(self, node: Optional[Node], key: int) -> Optional[Node]:
        """Рекурсивный помощник удаления узла по ключу."""
        if node is None:
            return None

        if key < node.key:
            node.left = self._delete_node(node.left, key)
        elif key > node.key:
            node.right = self._delete_node(node.right, key)
        else:
            # Узел найден: node.key == key
            # Случай 1: лист (оба потомка None) -> возвращаем None
            # Случай 2a: нет левого ребёнка -> возвращаем правого
            if node.left is None:
                return node.right
            # Случай 2b: нет правого ребёнка -> возвращаем левого
            if node.right is None:
                return node.left

            # Случай 3: узел имеет обоих детей.
            # Находим преемника — минимальный узел в правом поддереве.
            succ = self._min_node(node.right)
            node.key = succ.key
            # Удаляем узел преемника из правого поддерева
            node.right = self._delete_node(node.right, succ.key)

        return node

    @staticmethod
    def _min_node(node: Node) -> Node:
        """Поиск узла с наименьшим ключом в поддереве (крайний левый узел)."""
        curr = node
        while curr.left is not None:
            curr = curr.left
        return curr

    # -- Обходы --------------------------------------------------------------

    def in_order(self) -> List[int]:
        """Симметричный обход (L-N-R: левое поддерево -> корень -> правое): рекурсивно.

        Для корректного BST даёт строго возрастающую последовательность ключей.
        """
        res: List[int] = []

        def _dfs(node: Optional[Node]) -> None:
            if node is None:
                return
            _dfs(node.left)
            res.append(node.key)
            _dfs(node.right)

        _dfs(self.root)
        return res

    def pre_order(self) -> List[int]:
        """Прямой обход (N-L-R: корень -> левое поддерево -> правое): рекурсивно.

        Используется для сериализации структуры дерева и клонирования BST.
        """
        res: List[int] = []

        def _dfs(node: Optional[Node]) -> None:
            if node is None:
                return
            res.append(node.key)
            _dfs(node.left)
            _dfs(node.right)

        _dfs(self.root)
        return res

    def post_order(self) -> List[int]:
        """Обратный обход (L-R-N: левое поддерево -> правое -> корень): рекурсивно.

        Используется для освобождения ресурсов (bottom-up) и вычисления размеров.
        """
        res: List[int] = []

        def _dfs(node: Optional[Node]) -> None:
            if node is None:
                return
            _dfs(node.left)
            _dfs(node.right)
            res.append(node.key)

        _dfs(self.root)
        return res

    def level_order(self) -> List[int]:
        """Обход в ширину по уровням (BFS): итеративно через очередь (collections.deque).

        Обходит узлы уровень за уровнем от корня к листьям слева направо.
        """
        if self.root is None:
            return []

        res: List[int] = []
        queue: deque[Node] = deque([self.root])

        while queue:
            node = queue.popleft()
            res.append(node.key)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

        return res

    # -- Характеристики дерева ----------------------------------------------

    def height(self) -> int:
        """Высота дерева: максимальное число рёбер на пути от корня к листу.

        Пустое дерево: -1.
        Дерево из одного узла: 0.
        Реализовано итеративно (BFS) для защиты от переполнения стека.
        """
        if self.root is None:
            return -1

        queue: deque[Tuple[Node, int]] = deque([(self.root, 0)])
        max_h = 0
        while queue:
            node, depth = queue.popleft()
            if depth > max_h:
                max_h = depth
            if node.left is not None:
                queue.append((node.left, depth + 1))
            if node.right is not None:
                queue.append((node.right, depth + 1))

        return max_h

    def size(self) -> int:
        """Число узлов в дереве."""
        if self.root is None:
            return 0

        queue: deque[Node] = deque([self.root])
        count = 0
        while queue:
            node = queue.popleft()
            count += 1
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

        return count


# ---------------------------------------------------------------------------
# 2. Верификация: инвариант BST и сверка с эталоном (docs/ai-verification.md)
# ---------------------------------------------------------------------------


def self_check() -> None:
    """Полная верификация: граничные случаи, инвариант BST и сверка с эталоном."""
    print("Запуск self_check (граничные случаи и сверка с эталоном)...")

    # 1. Пустое дерево
    t = BST()
    assert t.search(1) is False
    assert t.in_order() == []
    assert t.pre_order() == []
    assert t.post_order() == []
    assert t.level_order() == []
    assert t.height() == -1 and t.size() == 0
    t.delete(1)  # удаление из пустого дерева не должно вызывать ошибок
    assert t.size() == 0 and t.height() == -1

    # 2. Дерево из одного узла
    t.insert(5)
    assert t.search(5) is True and t.search(4) is False
    assert t.height() == 0 and t.size() == 1
    assert t.in_order() == [5]
    assert t.pre_order() == [5]
    assert t.post_order() == [5]
    assert t.level_order() == [5]
    t.insert(5)  # повторная вставка игнорируется
    assert t.size() == 1

    # 3. Три случая удаления на фиксированном дереве: 5, 2, 8, 1, 3
    for k in (2, 8, 1, 3):
        t.insert(k)
    # Структура:
    #       5
    #      / \
    #     2   8
    #    / \
    #   1   3
    assert t.in_order() == [1, 2, 3, 5, 8]
    assert t.size() == 5

    # Случай 1: удаление листа (1)
    t.delete(1)
    assert t.in_order() == [2, 3, 5, 8]
    assert t.search(1) is False
    assert t.size() == 4

    # Случай 2: удаление узла с одним ребёнком (2 имеет только правого ребёнка 3)
    t.delete(2)
    assert t.in_order() == [3, 5, 8]
    assert t.search(2) is False
    assert t.size() == 3

    # Случай 3: удаление корня с двумя детьми (5 имеет детей 3 и 8, преемник = 8)
    t.delete(5)
    assert t.in_order() == [3, 8]
    assert t.search(5) is False
    assert t.size() == 2

    # 4. Обходы на каноническом дереве 4-2-6-1-3-5-7
    #         4
    #       /   \
    #      2     6
    #     / \   / \
    #    1   3 5   7
    t2 = BST()
    for k in (4, 2, 6, 1, 3, 5, 7):
        t2.insert(k)

    assert t2.in_order() == [1, 2, 3, 4, 5, 6, 7], "Ошибка in-order обхода"
    assert t2.pre_order() == [4, 2, 1, 3, 6, 5, 7], "Ошибка pre-order обхода"
    assert t2.post_order() == [1, 3, 2, 5, 7, 6, 4], "Ошибка post-order обхода"
    assert t2.level_order() == [4, 2, 6, 1, 3, 5, 7], "Ошибка level-order обхода"
    assert t2.height() == 2, f"Ожидалась высота 2, получено {t2.height()}"
    assert t2.size() == 7, f"Ожидался размер 7, получено {t2.size()}"

    # 5. Случайные последовательности команд: сверка с эталоном из stdlib (set + sorted)
    rng = random.Random(0)
    for trial in range(100):
        t3 = BST()
        etalon: set[int] = set()
        num_ops = rng.randint(1, 80)
        for _ in range(num_ops):
            k = rng.randint(-30, 30)
            if rng.random() < 0.6:
                t3.insert(k)
                etalon.add(k)
            else:
                t3.delete(k)
                etalon.discard(k)

            # Инвариант BST: in-order строго отсортирован и совпадает с эталоном
            in_order_keys = t3.in_order()
            expected_sorted = sorted(etalon)
            assert in_order_keys == expected_sorted, (
                f"Несовпадение in-order на шаге {trial}: {in_order_keys} != {expected_sorted}"
            )

        # Проверка поиска по всему диапазону
        for k in range(-30, 31):
            assert t3.search(k) == (k in etalon), f"Ошибка поиска ключа {k}"

        assert t3.size() == len(etalon), f"Несовпадение размера: {t3.size()} != {len(etalon)}"

    # 6. Тест деградации на упорядоченных данных (вырождение в список)
    deg_tree = BST()
    for i in range(15):
        deg_tree.insert(i)
    assert deg_tree.height() == 14, "Вырожденное дерево из 15 элементов должно иметь высоту 14"
    assert deg_tree.size() == 15
    assert deg_tree.in_order() == list(range(15))

    print("self_check: OK (все тесты успешно пройдены!)")


# ---------------------------------------------------------------------------
# 3. Бенчмарк: поиск в BST против поиска в списке (docs/reproducibility.md)
# ---------------------------------------------------------------------------

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
QUERIES = 1_000  # число поисков на один замер


def bench(fn: Callable[[], None]) -> float:
    """Медиана времени выполнения fn() по REPEATS запускам с 1 прогревочным прогоном."""
    fn()  # Прогрев интерпретатора и кэшей
    times: List[float] = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times)


def compute_slope(xs: List[float], ys: List[float]) -> float:
    """Вычисление наклона прямой в log-log координатах: log(y) = k * log(x) + b."""
    log_x = np.log(xs)
    log_y = np.log(ys)
    k, _ = np.polyfit(log_x, log_y, 1)
    return float(k)


def plot_results(
    sizes: List[int],
    heights: List[int],
    t_bst: List[float],
    t_list: List[float],
    variant: int,
) -> Path:
    """Построение комплексных графиков производительности и высоты дерева."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Log-Log график: Время поиска в BST и в Списке
    ax1 = axes[0, 0]
    ax1.plot(
        sizes,
        t_bst,
        "o-",
        color="#1f77b4",
        linewidth=2,
        markersize=6,
        label="BST: random keys, h = O(log n)",
    )
    ax1.plot(
        sizes,
        t_list,
        "s-",
        color="#d62728",
        linewidth=2,
        markersize=6,
        label="Список (list in): O(n)",
    )

    slope_bst = compute_slope(sizes, t_bst)
    slope_list = compute_slope(sizes, t_list)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Размер структуры n (число ключей)", fontsize=11)
    ax1.set_ylabel(f"Время {QUERIES} поисков, с (log)", fontsize=11)
    ax1.set_title(
        f"1. Время поиска: BST vs Список (Log-Log)\n"
        f"Наклон: BST k={slope_bst:.2f} (log-асимптотика); Список k={slope_list:.2f} (~O(n))",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=10, loc="upper left")

    # 2. Линейный график BST: Время как функция от log2(n)
    # (Требование docs/reproducibility.md для логарифмических алгоритмов)
    ax2 = axes[0, 1]
    log2_sizes = [math.log2(n) for n in sizes]
    ax2.plot(
        log2_sizes,
        t_bst,
        "o-",
        color="#2ca02c",
        linewidth=2,
        markersize=6,
        label="BST t(log₂ n)",
    )

    # Линейный тренд
    k_lin, b_lin = np.polyfit(log2_sizes, t_bst, 1)
    trend_y = [k_lin * x + b_lin for x in log2_sizes]
    ax2.plot(
        log2_sizes,
        trend_y,
        "--",
        color="gray",
        alpha=0.7,
        label=f"Линейная аппроксимация t = {k_lin:.2e}·log₂n + {b_lin:.2e}",
    )

    ax2.set_xlabel("log₂(n)", fontsize=11)
    ax2.set_ylabel(f"Время {QUERIES} поисков в BST, с", fontsize=11)
    ax2.set_title(
        "2. Время поиска в BST vs log₂(n) (линейные оси)\n"
        "Линейная зависимость подтверждает сложность O(log n)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="upper left")

    # 3. Высота BST в сравнении с теорией (AVL гарантия 1.44*log2(n) и log2(n))
    ax3 = axes[1, 0]
    ax3.plot(
        sizes,
        heights,
        "o-",
        color="#9467bd",
        linewidth=2,
        markersize=6,
        label="Эмпирическая высота BST: h(n)",
    )
    ideal_h = [math.log2(n) for n in sizes]
    avl_bound = [1.44 * math.log2(n) for n in sizes]
    random_expected = [2.0 * math.log(n) for n in sizes]  # ~ 1.386 * log2(n)

    ax3.plot(
        sizes,
        avl_bound,
        "--",
        color="#ff7f0e",
        linewidth=1.8,
        label="Верхняя граница AVL: 1.44·log₂(n)",
    )
    ax3.plot(
        sizes,
        random_expected,
        ":",
        color="#8c564b",
        linewidth=1.8,
        label="Теор. среднее случайного BST: 2·ln(n) ≈ 1.39·log₂(n)",
    )
    ax3.plot(
        sizes,
        ideal_h,
        "-.",
        color="gray",
        alpha=0.6,
        label="Идеально сбалансированное: log₂(n)",
    )

    ax3.set_xscale("log")
    ax3.set_xlabel("Число узлов n", fontsize=11)
    ax3.set_ylabel("Высота дерева (число рёбер)", fontsize=11)
    ax3.set_title(
        "3. Высота дерева BST: практика vs теория\n"
        "Случайные данные дают логарифмическую высоту близкую к AVL",
        fontsize=11,
        fontweight="bold",
    )
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=9, loc="upper left")

    # 4. Сводная карточка результатов и параметров
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary_text = (
        f"ЛАБОРАТОРНАЯ РАБОТА № 5\n"
        f"Бинарное дерево поиска (BST): операции и обходы\n"
        f"{'=' * 45}\n"
        f"• Студент: Данилов Всеволод Александрович\n"
        f"• Группа:  БВТ2503\n"
        f"• Вариант: {variant} (seed = {30 + variant})\n"
        f"• Число поисковых запросов: {QUERIES}\n"
        f"• Число повторов бенчмарка: {REPEATS} (медиана)\n"
        f"{'-' * 45}\n"
        f"Ключевые результаты:\n"
        f"1. Поиск в BST быстрее линейного поиска в списке:\n"
        f"   - при n = 1 000:   быстрее в ~{t_list[0] / t_bst[0]:.1f}x\n"
        f"   - при n = 100 000: быстрее в ~{t_list[-1] / t_bst[-1]:.1f}x\n\n"
        f"2. Асимптотика времени поиска:\n"
        f"   - Список: O(n), наклон log-log k = {slope_list:.2f}\n"
        f"   - BST:    O(h) = O(log n), график t vs log₂n линеен\n\n"
        f"3. Форма дерева и высота:\n"
        f"   - При n=100 000 высота BST составила h = {heights[-1]}\n"
        f"   - AVL-гарантия 1.44·log₂(100 000) ≈ {1.44 * math.log2(100000):.1f}\n"
        f"   - На упорядоченных ключах BST деградирует в список\n"
        f"     с высотой h = n - 1 и сложностью O(n)."
    )
    ax4.text(
        0.05,
        0.5,
        summary_text,
        fontsize=10.5,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=1.2", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )

    fig.suptitle(
        f"ЛР 5. Исследование бинарного дерева поиска (BST) — Вариант {variant}",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()

    out_path = Path(__file__).resolve().with_name(f"lab05_graphs_variant{variant}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Графики сохранены: {out_path.name}")
    return out_path


def run_benchmarks(seed: int, variant: int) -> None:
    """Запуск бенчмарка: сравнение времени поиска в BST и в списке."""
    print(f"\n=== БЕНЧМАРК (Вариант {variant}, Seed = {seed}) ===")
    rng = random.Random(seed)

    heights: List[int] = []
    t_bst_list: List[float] = []
    t_list_list: List[float] = []

    header = f"{'n':>8} {'h(BST)':>8} {'t_bst, c':>12} {'t_list, c':>12} {'Ускорение':>12}"
    print(header)
    print("-" * len(header))

    for n in SIZES:
        keys = rng.sample(range(10 * n), n)
        queries = [rng.randint(0, 10 * n) for _ in range(QUERIES)]

        tree = BST()
        for k in keys:
            tree.insert(k)

        h = tree.height()
        heights.append(h)

        def search_bst() -> None:
            for q in queries:
                tree.search(q)

        def search_list() -> None:
            for q in queries:
                _ = q in keys

        t_bst = bench(search_bst)
        t_list = bench(search_list)

        t_bst_list.append(t_bst)
        t_list_list.append(t_list)

        speedup = t_list / t_bst if t_bst > 0 else 0.0
        print(f"{n:>8} {h:>8} {t_bst:>12.6f} {t_list:>12.6f} {speedup:>11.1f}x")

    # Деградация на упорядоченных данных
    print("\n--- Демонстрация деградации BST на упорядоченных ключах (h = n - 1) ---")
    deg_sizes = [500, 1000, 2000]
    for dn in deg_sizes:
        deg_keys = list(range(dn))
        deg_tree = BST()
        for k in deg_keys:
            deg_tree.insert(k)
        deg_queries = [rng.randint(0, dn) for _ in range(QUERIES)]

        def search_deg_bst() -> None:
            for q in deg_queries:
                deg_tree.search(q)

        t_deg = bench(search_deg_bst)
        print(f"Вырожденное BST n={dn:>5}: высота h={deg_tree.height():>5}, время {QUERIES} поисков = {t_deg:.6f} c")

    # Построение графиков
    plot_results(SIZES, heights, t_bst_list, t_list_list, variant)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", type=int, required=True, help="номер варианта (для seed = 30 + variant)")
    args = ap.parse_args()

    seed = 30 + args.variant
    random.seed(seed)

    self_check()
    run_benchmarks(seed, args.variant)


if __name__ == "__main__":
    main()
