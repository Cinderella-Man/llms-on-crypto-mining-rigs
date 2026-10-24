# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/zipper/canonical-data.json
# File last updated on 2023-07-19

import unittest

from zipper import (
    Zipper,
)


class ZipperTest(unittest.TestCase):
    def test_data_is_retained(self):
        initial = {
            "value": 1,
            "left": {
                "value": 2,
                "left": None,
                "right": {"value": 3, "left": None, "right": None},
            },
            "right": {"value": 4, "left": None, "right": None},
        }

        expected = {
            "value": 1,
            "left": {
                "value": 2,
                "left": None,
                "right": {"value": 3, "left": None, "right": None},
            },
            "right": {"value": 4, "left": None, "right": None},
        }

        zipper = Zipper.from_tree(initial)
        result = zipper.to_tree()
        self.assertEqual(result, expected)

    def test_tree_from_deep_focus(self):
        initial = {
            "value": 1,
            "left": {
                "value": 2,
                "left": None,
                "right": {"value": 3, "left": None, "right": None},
            },
            "right": {"value": 4, "left": None, "right": None},
        }

        expected = {
            "value": 1,
            "left": {
                "value": 2,
                "left": None,
                "right": {"value": 3, "left": None, "right": None},
            },
            "right": {"value": 4, "left": None, "right": None},
        }

        zipper = Zipper.from_tree(initial)
        result = zipper.left().right().to_tree()
        self.assertEqual(result, expected)

    def test_test_ability_to_descend_multiple_levels_and_return(self):
        initial = {
            "value": 1,
            "left": {
                "value": 2,
                "left": None,
                "right": {"value": 3, "left": None, "right": None},
            },
            "right": {"value": 4, "left": None, "right": None},
        }

        zipper = Zipper.from_tree(initial)
        result = zipper.left().right().up().up().value()
        self.assertEqual(result, 1)

    def test_set_left_with_leaf(self):
        initial = {
            "value": 1,
            "left": {
                "value": 2,
                "left": None,
                "right": {"value": 3, "left": None, "right": None},
            },
            "right": {"value": 4, "left": None, "right": None},
        }

        expected = {
            "value": 1,
            "left": {
                "value": 2,
                "left": {"value": 5, "left": None, "right": None},
                "right": {"value": 3, "left": None, "right": None},
            },
            "right": {"value": 4, "left": None, "right": None},
        }

        zipper = Zipper.from_tree(initial)
        result = (
            zipper.left().set_left({"value": 5, "left": None, "right": None}).to_tree()
        )
        self.assertEqual(result, expected)

    def test_set_value_on_deep_focus(self):
        initial = {
            "value": 1,
            "left": {
                "value": 2,
                "left": None,
                "right": {"value": 3, "left": None, "right": None},
            },
            "right": {"value": 4, "left": None, "right": None},
        }

        expected = {
            "value": 1,
            "left": {
                "value": 2,
                "left": None,
                "right": {"value": 5, "left": None, "right": None},
            },
            "right": {"value": 4, "left": None, "right": None},
        }

        zipper = Zipper.from_tree(initial)
        result = zipper.left().right().set_value(5).to_tree()
        self.assertEqual(result, expected)
