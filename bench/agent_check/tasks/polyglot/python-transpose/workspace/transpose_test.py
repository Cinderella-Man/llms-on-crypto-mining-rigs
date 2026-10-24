# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/transpose/canonical-data.json
# File last updated on 2024-08-26

import unittest

from transpose import (
    transpose,
)


class TransposeTest(unittest.TestCase):
    def test_empty_string(self):
        text = ""
        expected = ""

        self.assertEqual(transpose(text), expected)

    def test_simple(self):
        text = "ABC\n123"
        expected = "A1\nB2\nC3"

        self.assertEqual(transpose(text), expected)

    def test_second_line_longer_than_first_line(self):
        text = "The first line.\nThe second line."
        expected = "TT\nhh\nee\n  \nfs\nie\nrc\nso\ntn\n d\nl \nil\nni\nen\n.e\n ."

        self.assertEqual(transpose(text), expected)

    def test_rectangle(self):
        text = "FRACTURE\nOUTLINED\nBLOOMING\nSEPTETTE"
        expected = "FOBS\nRULE\nATOP\nCLOT\nTIME\nUNIT\nRENT\nEDGE"

        self.assertEqual(transpose(text), expected)
