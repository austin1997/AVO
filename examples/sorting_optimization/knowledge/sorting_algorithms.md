# Sorting Algorithm Reference

## Common Sorting Algorithms and Their Complexities

### Comparison-based sorts

| Algorithm      | Best     | Average  | Worst    | Space  | Stable |
|---------------|----------|----------|----------|--------|--------|
| Bubble Sort   | O(n)     | O(n^2)   | O(n^2)   | O(1)   | Yes    |
| Insertion Sort| O(n)     | O(n^2)   | O(n^2)   | O(1)   | Yes    |
| Selection Sort| O(n^2)   | O(n^2)   | O(n^2)   | O(1)   | No     |
| Merge Sort    | O(nlogn) | O(nlogn) | O(nlogn) | O(n)   | Yes    |
| Quick Sort    | O(nlogn) | O(nlogn) | O(n^2)   | O(logn)| No     |
| Heap Sort     | O(nlogn) | O(nlogn) | O(nlogn) | O(1)   | No     |
| Tim Sort      | O(n)     | O(nlogn) | O(nlogn) | O(n)   | Yes    |

### Non-comparison sorts (integers only)

| Algorithm      | Time     | Space  |
|---------------|----------|--------|
| Counting Sort | O(n+k)   | O(k)   |
| Radix Sort    | O(d(n+k))| O(n+k) |
| Bucket Sort   | O(n+k)   | O(n)   |

## Python-Specific Optimization Tips

1. **Minimize attribute lookups**: Use local variables for frequently accessed items.
2. **Use list comprehensions** over manual loops when possible.
3. **Avoid unnecessary copies**: Only copy when the original must be preserved.
4. **Use built-in functions**: `sorted()` and `list.sort()` use TimSort (C implementation).
5. **For integer arrays**: Consider radix sort or counting sort for O(n) performance.
6. **Hybrid approaches**: Use insertion sort for small subarrays within quicksort/mergesort.
7. **Three-way partitioning**: Handles duplicate elements efficiently in quicksort.

## Implementation Notes

- The solution must define `sort_array(arr: list[int]) -> list[int]`
- It must return a new sorted list (not sort in-place)
- Must handle edge cases: empty arrays, single elements, already sorted, reverse sorted
- Integer values range from -100000 to 100000
