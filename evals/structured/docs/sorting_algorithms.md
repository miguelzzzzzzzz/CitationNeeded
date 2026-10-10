---
title: Comparison Sorting Algorithms
topic: algorithms
---

# Comparison Sorting Algorithms

## Quicksort

Quicksort partitions around a pivot and recurses on both sides. Average time
is O(n log n); worst case O(n^2) with adversarial pivots.

## Mergesort

Mergesort splits the array, sorts each half, then merges. It guarantees
O(n log n) time and uses O(n) auxiliary memory.

## Heapsort

Heapsort builds a max-heap then repeatedly extracts the maximum. It runs in
O(n log n) time with O(1) extra memory.
