# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/forth/canonical-data.json
# File last updated on 2026-02-19

import unittest

from forth import (
    evaluate,
    StackUnderflowError,
)


class ForthTest(unittest.TestCase):

    def test_parsing_and_numbers_numbers_just_get_pushed_onto_the_stack(self):
        self.assertEqual(evaluate(["1 2 3 4 5"]), [1, 2, 3, 4, 5])

    def test_addition_errors_if_there_is_nothing_on_the_stack(self):
        with self.assertRaises(StackUnderflowError) as err:
            evaluate(["+"])
        self.assertEqual(type(err.exception), StackUnderflowError)
        self.assertEqual(
            str(err.exception.args[0]), "Insufficient number of items in stack"
        )

    def test_subtraction_can_subtract_two_numbers(self):
        self.assertEqual(evaluate(["3 4 -"]), [-1])

    def test_subtraction_more_than_two_values_on_the_stack(self):
        self.assertEqual(evaluate(["1 12 3 -"]), [1, 9])

    def test_multiplication_errors_if_there_is_only_one_value_on_the_stack(self):
        with self.assertRaises(StackUnderflowError) as err:
            evaluate(["1 *"])
        self.assertEqual(type(err.exception), StackUnderflowError)
        self.assertEqual(
            str(err.exception.args[0]), "Insufficient number of items in stack"
        )

    def test_division_performs_integer_division(self):
        self.assertEqual(evaluate(["8 3 /"]), [2])

    def test_division_errors_if_there_is_only_one_value_on_the_stack(self):
        with self.assertRaises(StackUnderflowError) as err:
            evaluate(["1 /"])
        self.assertEqual(type(err.exception), StackUnderflowError)
        self.assertEqual(
            str(err.exception.args[0]), "Insufficient number of items in stack"
        )

    def test_combined_arithmetic_multiplication_and_division(self):
        self.assertEqual(evaluate(["2 4 * 3 /"]), [2])

    def test_dup_copies_a_value_on_the_stack(self):
        self.assertEqual(evaluate(["1 dup"]), [1, 1])

    def test_drop_removes_the_top_value_on_the_stack_if_it_is_the_only_one(self):
        self.assertEqual(evaluate(["1 drop"]), [])

    def test_swap_swaps_the_top_two_values_on_the_stack_if_they_are_the_only_ones(self):
        self.assertEqual(evaluate(["1 2 swap"]), [2, 1])

    def test_swap_errors_if_there_is_only_one_value_on_the_stack(self):
        with self.assertRaises(StackUnderflowError) as err:
            evaluate(["1 swap"])
        self.assertEqual(type(err.exception), StackUnderflowError)
        self.assertEqual(
            str(err.exception.args[0]), "Insufficient number of items in stack"
        )

    def test_over_errors_if_there_is_nothing_on_the_stack(self):
        with self.assertRaises(StackUnderflowError) as err:
            evaluate(["over"])
        self.assertEqual(type(err.exception), StackUnderflowError)
        self.assertEqual(
            str(err.exception.args[0]), "Insufficient number of items in stack"
        )

    def test_user_defined_words_execute_in_the_right_order(self):
        self.assertEqual(evaluate([": countup 1 2 3 ;", "countup"]), [1, 2, 3])

    def test_user_defined_words_can_override_built_in_operators(self):
        self.assertEqual(evaluate([": + * ;", "3 4 +"]), [12])

    def test_user_defined_words_cannot_redefine_non_negative_numbers(self):
        with self.assertRaises(ValueError) as err:
            evaluate([": 1 2 ;"])
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(str(err.exception.args[0]), "illegal operation")

    def test_case_insensitivity_dup_is_case_insensitive(self):
        self.assertEqual(evaluate(["1 DUP Dup dup"]), [1, 1, 1, 1])

    def test_case_insensitivity_over_is_case_insensitive(self):
        self.assertEqual(evaluate(["1 2 OVER Over over"]), [1, 2, 1, 2, 1])
