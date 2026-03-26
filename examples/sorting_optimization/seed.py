"""Seed sorting implementation: naive bubble sort.

This is the starting point (x0) for AVO evolution.
The agent will iteratively optimize this into faster sorting algorithms.
"""


def sort_array(arr: list[int]) -> list[int]:
    """Sort an array of integers in ascending order."""
    result = arr.copy()
    n = len(result)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result
