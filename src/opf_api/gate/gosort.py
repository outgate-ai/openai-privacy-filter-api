# Vendored from outgate-ai/opf-bench src/opfbench/gosort.py (0976e55); keep in sync, do not edit here.
"""Go's sort.Slice (pdqsort_func, src/sort/zsortfunc.go, unchanged go1.19-1.25).

sort.Slice is not stable, and the gate's mergeSpans keeps the first of several
spans with equal bounds, so matching the gate needs Go's exact element order.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")
_U64 = (1 << 64) - 1


class _Data:
    __slots__ = ("xs", "lt")

    def __init__(self, xs: list, lt: Callable):
        self.xs = xs
        self.lt = lt

    def less(self, i: int, j: int) -> bool:
        return self.lt(self.xs[i], self.xs[j])

    def swap(self, i: int, j: int) -> None:
        xs = self.xs
        xs[i], xs[j] = xs[j], xs[i]


def sort_slice(xs: list[T], less: Callable[[T, T], bool]) -> None:
    n = len(xs)
    _pdqsort(_Data(xs, less), 0, n, n.bit_length())


def _insertion(d: _Data, a: int, b: int) -> None:
    for i in range(a + 1, b):
        j = i
        while j > a and d.less(j, j - 1):
            d.swap(j, j - 1)
            j -= 1


def _sift_down(d: _Data, lo: int, hi: int, first: int) -> None:
    root = lo
    while True:
        child = 2 * root + 1
        if child >= hi:
            return
        if child + 1 < hi and d.less(first + child, first + child + 1):
            child += 1
        if not d.less(first + root, first + child):
            return
        d.swap(first + root, first + child)
        root = child


def _heap(d: _Data, a: int, b: int) -> None:
    first, lo, hi = a, 0, b - a
    for i in range((hi - 1) // 2, -1, -1):
        _sift_down(d, i, hi, first)
    for i in range(hi - 1, -1, -1):
        d.swap(first, first + i)
        _sift_down(d, lo, i, first)


_UNKNOWN, _INCREASING, _DECREASING = 0, 1, 2


def _pdqsort(d: _Data, a: int, b: int, limit: int) -> None:
    max_insertion = 12
    was_balanced = True
    was_partitioned = True
    while True:
        length = b - a
        if length <= max_insertion:
            _insertion(d, a, b)
            return
        if limit == 0:
            _heap(d, a, b)
            return
        if not was_balanced:
            _break_patterns(d, a, b)
            limit -= 1
        pivot, hint = _choose_pivot(d, a, b)
        if hint == _DECREASING:
            _reverse(d, a, b)
            pivot = (b - 1) - (pivot - a)
            hint = _INCREASING
        if was_balanced and was_partitioned and hint == _INCREASING:
            if _partial_insertion(d, a, b):
                return
        if a > 0 and not d.less(a - 1, pivot):
            a = _partition_equal(d, a, b, pivot)
            continue
        mid, already = _partition(d, a, b, pivot)
        was_partitioned = already
        left, right = mid - a, b - mid
        threshold = length // 8
        if left < right:
            was_balanced = left >= threshold
            _pdqsort(d, a, mid, limit)
            a = mid + 1
        else:
            was_balanced = right >= threshold
            _pdqsort(d, mid + 1, b, limit)
            b = mid


def _partition(d: _Data, a: int, b: int, pivot: int) -> tuple[int, bool]:
    d.swap(a, pivot)
    i, j = a + 1, b - 1
    while i <= j and d.less(i, a):
        i += 1
    while i <= j and not d.less(j, a):
        j -= 1
    if i > j:
        d.swap(j, a)
        return j, True
    d.swap(i, j)
    i += 1
    j -= 1
    while True:
        while i <= j and d.less(i, a):
            i += 1
        while i <= j and not d.less(j, a):
            j -= 1
        if i > j:
            break
        d.swap(i, j)
        i += 1
        j -= 1
    d.swap(j, a)
    return j, False


def _partition_equal(d: _Data, a: int, b: int, pivot: int) -> int:
    d.swap(a, pivot)
    i, j = a + 1, b - 1
    while True:
        while i <= j and not d.less(a, i):
            i += 1
        while i <= j and d.less(a, j):
            j -= 1
        if i > j:
            break
        d.swap(i, j)
        i += 1
        j -= 1
    return i


def _partial_insertion(d: _Data, a: int, b: int) -> bool:
    max_steps, shortest_shifting = 5, 50
    i = a + 1
    for _ in range(max_steps):
        while i < b and not d.less(i, i - 1):
            i += 1
        if i == b:
            return True
        if b - a < shortest_shifting:
            return False
        d.swap(i, i - 1)
        if i - a >= 2:
            j = i - 1
            while j >= 1:
                if not d.less(j, j - 1):
                    break
                d.swap(j, j - 1)
                j -= 1
        if b - i >= 2:
            j = i + 1
            while j < b:
                if not d.less(j, j - 1):
                    break
                d.swap(j, j - 1)
                j += 1
    return False


def _break_patterns(d: _Data, a: int, b: int) -> None:
    length = b - a
    if length < 8:
        return
    r = length & _U64
    modulus = 1 << length.bit_length()
    idx = a + (length // 4) * 2 - 1
    while idx <= a + (length // 4) * 2 + 1:
        r ^= (r << 13) & _U64
        r ^= r >> 17
        r ^= (r << 5) & _U64
        other = r & (modulus - 1)
        if other >= length:
            other -= length
        d.swap(idx, a + other)
        idx += 1


def _choose_pivot(d: _Data, a: int, b: int) -> tuple[int, int]:
    shortest_ninther, max_swaps = 50, 4 * 3
    n = b - a
    swaps = [0]
    i = a + n // 4 * 1
    j = a + n // 4 * 2
    k = a + n // 4 * 3
    if n >= 8:
        if n >= shortest_ninther:
            i = _median_adjacent(d, i, swaps)
            j = _median_adjacent(d, j, swaps)
            k = _median_adjacent(d, k, swaps)
        j = _median(d, i, j, k, swaps)
    if swaps[0] == 0:
        return j, _INCREASING
    if swaps[0] == max_swaps:
        return j, _DECREASING
    return j, _UNKNOWN


def _order2(d: _Data, a: int, b: int, swaps: list[int]) -> tuple[int, int]:
    if d.less(b, a):
        swaps[0] += 1
        return b, a
    return a, b


def _median(d: _Data, a: int, b: int, c: int, swaps: list[int]) -> int:
    a, b = _order2(d, a, b, swaps)
    b, c = _order2(d, b, c, swaps)
    a, b = _order2(d, a, b, swaps)
    return b


def _median_adjacent(d: _Data, a: int, swaps: list[int]) -> int:
    return _median(d, a - 1, a, a + 1, swaps)


def _reverse(d: _Data, a: int, b: int) -> None:
    i, j = a, b - 1
    while i < j:
        d.swap(i, j)
        i += 1
        j -= 1
