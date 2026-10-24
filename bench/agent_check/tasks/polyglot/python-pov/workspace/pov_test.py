# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/pov/canonical-data.json
# File last updated on 2026-02-19

import unittest

from pov import (
    Tree,
)


class PovTest(unittest.TestCase):

    def test_results_in_the_same_tree_if_the_input_tree_is_a_singleton(self):
        tree = Tree("x")
        expected = Tree("x")
        self.assertTreeEquals(tree.from_pov("x"), expected)

    def test_can_reroot_a_tree_with_new_root_deeply_nested_in_tree(self):
        tree = Tree(
            "level-0",
            [Tree("level-1", [Tree("level-2", [Tree("level-3", [Tree("x")])])])],
        )
        expected = Tree(
            "x",
            [Tree("level-3", [Tree("level-2", [Tree("level-1", [Tree("level-0")])])])],
        )
        self.assertTreeEquals(tree.from_pov("x"), expected)

    def test_errors_if_target_does_not_exist_in_a_singleton_tree(self):
        tree = Tree("x")
        with self.assertRaises(ValueError) as err:
            tree.from_pov("nonexistent")
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "Tree could not be reoriented")

    def test_can_find_path_to_sibling(self):
        tree = Tree("parent", [Tree("a"), Tree("x"), Tree("b"), Tree("c")])
        expected = ["x", "parent", "b"]
        self.assertEqual(tree.path_to("x", "b"), expected)

    def test_can_find_path_from_nodes_other_than_x(self):
        tree = Tree("parent", [Tree("a"), Tree("x"), Tree("b"), Tree("c")])
        expected = ["a", "parent", "c"]
        self.assertEqual(tree.path_to("a", "c"), expected)

    # Custom Utility Functions
    def assertTreeEquals(self, result, expected):
        self.assertEqual(result, expected, "{} != {}".format(result, expected))
