# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/wordy/canonical-data.json
# File last updated on 2026-05-22

import unittest

from wordy import (
    answer,
)


class WordyTest(unittest.TestCase):
    def test_just_a_number(self):
        self.assertEqual(answer("What is 5?"), 5)

    def test_addition(self):
        self.assertEqual(answer("What is 1 plus 1?"), 2)

    def test_more_addition(self):
        self.assertEqual(answer("What is 53 plus 2?"), 55)

    def test_subtraction(self):
        self.assertEqual(answer("What is 4 minus -12?"), 16)

    def test_multiple_additions(self):
        self.assertEqual(answer("What is 1 plus 1 plus 1?"), 3)

    def test_subtraction_then_addition(self):
        self.assertEqual(answer("What is 17 minus 6 plus 3?"), 14)

    def test_multiple_division(self):
        self.assertEqual(answer("What is -12 divided by 2 divided by -3?"), 2)

    def test_reject_problem_with_no_operands_or_operators(self):
        with self.assertRaises(ValueError) as err:
            answer("What is?")
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "syntax error")

    def test_reject_postfix_notation(self):
        with self.assertRaises(ValueError) as err:
            answer("What is 1 2 plus?")
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "syntax error")

    # Additional tests for this track

    def test_4_number_question_with_zero(self):
        self.assertEqual(answer("What is 12 minus 0 divided by 6 plus 5?"), 7)

    def test_missing_number(self):
        with self.assertRaises(ValueError) as err:
            answer("What is 7 plus multiplied by -2?")
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "syntax error")
