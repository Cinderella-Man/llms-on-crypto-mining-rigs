# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/two-bucket/canonical-data.json
# File last updated on 2025-09-23

import unittest

from two_bucket import (
    measure,
)


class TwoBucketTest(unittest.TestCase):
    def test_measure_using_bucket_one_of_size_3_and_bucket_two_of_size_5_start_with_bucket_one(
        self,
    ):
        self.assertEqual(measure(3, 5, 1, "one"), (4, "one", 5))

    def test_measure_using_bucket_one_of_size_7_and_bucket_two_of_size_11_start_with_bucket_two(
        self,
    ):
        self.assertEqual(measure(7, 11, 2, "two"), (18, "two", 7))

    def test_measure_using_bucket_one_much_bigger_than_bucket_two(self):
        self.assertEqual(measure(5, 1, 2, "one"), (6, "one", 1))

    def test_with_the_same_buckets_but_a_different_goal_then_it_is_possible(self):
        self.assertEqual(measure(6, 15, 9, "one"), (10, "two", 0))

    # Utility functions
    def assertRaisesWithMessage(self, exception):
        return self.assertRaisesRegex(exception, r".+")
