# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/affine-cipher/canonical-data.json
# File last updated on 2026-02-19

import unittest

from affine_cipher import (
    decode,
    encode,
)


class AffineCipherTest(unittest.TestCase):

    def test_encode_yes(self):
        self.assertEqual(encode("yes", 5, 7), "xbt")

    def test_encode_o_m_g(self):
        self.assertEqual(encode("O M G", 25, 47), "hjp")

    def test_encode_deep_thought(self):
        self.assertEqual(encode("Truth is fiction.", 5, 17), "iynia fdqfb ifje")

    def test_decode_exercism(self):
        self.assertEqual(decode("tytgn fjr", 3, 7), "exercism")

    def test_decode_all_the_letters(self):
        self.assertEqual(
            decode("swxtj npvyk lruol iejdc blaxk swxmh qzglf", 17, 33),
            "thequickbrownfoxjumpsoverthelazydog",
        )

    def test_decode_with_a_not_coprime_to_m(self):
        with self.assertRaises(ValueError) as err:
            decode("Test", 13, 5)
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "a and m must be coprime.")
