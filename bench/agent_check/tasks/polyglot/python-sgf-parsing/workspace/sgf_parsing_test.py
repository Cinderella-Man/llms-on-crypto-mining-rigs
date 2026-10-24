# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/sgf-parsing/canonical-data.json
# File last updated on 2026-02-19

import unittest

from sgf_parsing import (
    parse,
    SgfTree,
)


class SgfParsingTest(unittest.TestCase):

    def test_empty_input(self):
        input_string = ""
        with self.assertRaises(ValueError) as err:
            parse(input_string)
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "tree missing")

    def test_node_without_properties(self):
        input_string = "(;)"
        expected = SgfTree()
        self.assertEqual(parse(input_string), expected)

    def test_properties_without_delimiter(self):
        input_string = "(;A)"
        with self.assertRaises(ValueError) as err:
            parse(input_string)
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "properties without delimiter")

    def test_two_nodes(self):
        input_string = "(;A[B];B[C])"
        expected = SgfTree(properties={"A": ["B"]}, children=[SgfTree({"B": ["C"]})])
        self.assertEqual(parse(input_string), expected)

    def test_within_property_values_whitespace_characters_such_as_tab_are_converted_to_spaces(
        self,
    ):
        input_string = "(;A[hello\t\tworld])"
        expected = SgfTree(properties={"A": ["hello  world"]})
        self.assertEqual(parse(input_string), expected)

    def test_escaped_backslash_in_property_value_becomes_just_a_backslash(self):
        input_string = "(;A[\\\\])"
        expected = SgfTree(properties={"A": ["\\"]})
        self.assertEqual(parse(input_string), expected)

    def test_parentheses_in_property_value_don_t_need_to_be_escaped(self):
        input_string = "(;A[x(y)z][foo]B[bar];C[baz])"
        expected = SgfTree(
            properties={"A": ["x(y)z", "foo"], "B": ["bar"]},
            children=[SgfTree({"C": ["baz"]})],
        )
        self.assertEqual(parse(input_string), expected)

    def test_escaped_t_and_n_in_property_value_are_just_letters_not_whitespace(self):
        input_string = "(;A[\\t = t and \\n = n])"
        expected = SgfTree(properties={"A": ["t = t and n = n"]})
        self.assertEqual(parse(input_string), expected)
