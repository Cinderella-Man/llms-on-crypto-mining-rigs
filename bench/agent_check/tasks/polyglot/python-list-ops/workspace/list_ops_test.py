# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/list-ops/canonical-data.json
# File last updated on 2026-09-29

import unittest

from list_ops import (
    append,
    concat,
    foldl,
    foldr,
    length,
    reverse,
    filter as list_ops_filter,
    map as list_ops_map,
)


class ListOpsTest(unittest.TestCase):
    def test_append_empty_lists(self):
        self.assertEqual(append([], []), [])

    def test_append_non_empty_lists(self):
        self.assertEqual(append([1, 2], [2, 3, 4, 5]), [1, 2, 2, 3, 4, 5])

    def test_concat_list_of_nested_lists(self):
        self.assertEqual(
            concat([[[1], [2]], [[3]], [[]], [[4, 5, 6]]]),
            [[1], [2], [3], [], [4, 5, 6]],
        )

    def test_length_empty_list(self):
        self.assertEqual(length([]), 0)

    def test_map_non_empty_list(self):
        self.assertEqual(list_ops_map(lambda x: x + 1, [1, 3, 5, 7]), [2, 4, 6, 8])

    def test_foldl_direction_dependent_function_applied_to_non_empty_list(self):
        self.assertEqual(foldl(lambda acc, el: el / acc, [1, 2, 3, 4], 24), 64)

    def test_foldr_direction_dependent_function_applied_to_non_empty_list(self):
        self.assertEqual(foldr(lambda acc, el: el / acc, [1, 2, 3, 4], 24), 9)

    def test_reverse_non_empty_odd_length_list(self):
        self.assertEqual(reverse([1, 3, 5, 7, 9, 11, 13]), [13, 11, 9, 7, 5, 3, 1])

    # Additional tests for this track

    def test_reverse_reverse_mixed_types(self):
        self.assertEqual(reverse(["xyz", 4.0, "cat", 1]), [1, "cat", 4.0, "xyz"])
