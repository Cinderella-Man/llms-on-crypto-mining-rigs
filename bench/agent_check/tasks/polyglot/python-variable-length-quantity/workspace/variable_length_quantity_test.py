# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/variable-length-quantity/canonical-data.json
# File last updated on 2026-02-19

import unittest

from variable_length_quantity import (
    decode,
    encode,
)


class VariableLengthQuantityTest(unittest.TestCase):

    def test_zero(self):
        self.assertEqual(encode([0x0]), [0x0])

    def test_largest_single_byte(self):
        self.assertEqual(encode([0x7F]), [0x7F])

    def test_asymmetric_double_byte(self):
        self.assertEqual(encode([0xAD]), [0x81, 0x2D])

    def test_arbitrary_triple_byte(self):
        self.assertEqual(encode([0x100000]), [0xC0, 0x80, 0x0])

    def test_smallest_quadruple_byte(self):
        self.assertEqual(encode([0x200000]), [0x81, 0x80, 0x80, 0x0])

    def test_largest_quadruple_byte(self):
        self.assertEqual(encode([0xFFFFFFF]), [0xFF, 0xFF, 0xFF, 0x7F])

    def test_asymmetric_quintuple_byte(self):
        self.assertEqual(encode([0x86656105]), [0x88, 0xB3, 0x95, 0xC2, 0x5])

    def test_two_multi_byte_values(self):
        self.assertEqual(
            encode([0x4000, 0x123456]), [0x81, 0x80, 0x0, 0xC8, 0xE8, 0x56]
        )

    def test_two_bytes(self):
        self.assertEqual(decode([0xC0, 0x0]), [0x2000])

    def test_maximum_32_bit_integer(self):
        self.assertEqual(decode([0x8F, 0xFF, 0xFF, 0xFF, 0x7F]), [0xFFFFFFFF])

    def test_multiple_values(self):
        self.assertEqual(
            decode(
                [
                    0xC0,
                    0x0,
                    0xC8,
                    0xE8,
                    0x56,
                    0xFF,
                    0xFF,
                    0xFF,
                    0x7F,
                    0x0,
                    0xFF,
                    0x7F,
                    0x81,
                    0x80,
                    0x0,
                ]
            ),
            [0x2000, 0x123456, 0xFFFFFFF, 0x0, 0x3FFF, 0x4000],
        )
