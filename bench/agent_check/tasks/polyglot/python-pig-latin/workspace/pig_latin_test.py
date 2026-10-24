# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/pig-latin/canonical-data.json
# File last updated on 2026-02-19

import unittest

from pig_latin import (
    translate,
)


class PigLatinTest(unittest.TestCase):

    def test_word_beginning_with_a(self):
        self.assertEqual(translate("apple"), "appleay")

    def test_word_beginning_with_o(self):
        self.assertEqual(translate("object"), "objectay")

    def test_word_beginning_with_p(self):
        self.assertEqual(translate("pig"), "igpay")

    def test_word_beginning_with_q_without_a_following_u(self):
        self.assertEqual(translate("qat"), "atqay")

    def test_word_beginning_with_qu(self):
        self.assertEqual(translate("queen"), "eenquay")

    def test_word_beginning_with_thr(self):
        self.assertEqual(translate("thrush"), "ushthray")

    def test_word_beginning_with_xr(self):
        self.assertEqual(translate("xray"), "xrayay")

    def test_y_as_second_letter_in_two_letter_word(self):
        self.assertEqual(translate("my"), "ymay")
