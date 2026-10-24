# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/phone-number/canonical-data.json
# File last updated on 2023-07-19

import unittest

from phone_number import (
    PhoneNumber,
)


class PhoneNumberTest(unittest.TestCase):
    def test_cleans_the_number(self):
        number = PhoneNumber("(223) 456-7890").number
        self.assertEqual(number, "2234567890")

    def test_invalid_when_9_digits(self):
        with self.assertRaises(ValueError) as err:
            PhoneNumber("123456789")
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "must not be fewer than 10 digits")

    def test_valid_when_11_digits_and_starting_with_1_even_with_punctuation(self):
        number = PhoneNumber("+1 (223) 456-7890").number
        self.assertEqual(number, "2234567890")

    def test_invalid_with_punctuations(self):
        with self.assertRaises(ValueError) as err:
            PhoneNumber("523-@:!-7890")
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "punctuations not permitted")

    def test_invalid_if_exchange_code_starts_with_0(self):
        with self.assertRaises(ValueError) as err:
            PhoneNumber("(223) 056-7890")
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "exchange code cannot start with zero")

    def test_invalid_if_area_code_starts_with_1_on_valid_11_digit_number(self):
        with self.assertRaises(ValueError) as err:
            PhoneNumber("1 (123) 456-7890")
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "area code cannot start with one")

    # Additional tests for this track
    def test_area_code(self):
        number = PhoneNumber("2234567890")
        self.assertEqual(number.area_code, "223")
