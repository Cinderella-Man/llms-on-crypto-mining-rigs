# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/beer-song/canonical-data.json
# File last updated on 2023-07-20

import unittest

from beer_song import (
    recite,
)


class BeerSongTest(unittest.TestCase):
    def test_first_generic_verse(self):
        expected = [
            "99 bottles of beer on the wall, 99 bottles of beer.",
            "Take one down and pass it around, 98 bottles of beer on the wall.",
        ]
        self.assertEqual(recite(start=99), expected)

    def test_verse_with_1_bottle(self):
        expected = [
            "1 bottle of beer on the wall, 1 bottle of beer.",
            "Take it down and pass it around, no more bottles of beer on the wall.",
        ]
        self.assertEqual(recite(start=1), expected)

    def test_last_three_verses(self):
        expected = [
            "2 bottles of beer on the wall, 2 bottles of beer.",
            "Take one down and pass it around, 1 bottle of beer on the wall.",
            "",
            "1 bottle of beer on the wall, 1 bottle of beer.",
            "Take it down and pass it around, no more bottles of beer on the wall.",
            "",
            "No more bottles of beer on the wall, no more bottles of beer.",
            "Go to the store and buy some more, 99 bottles of beer on the wall.",
        ]
        self.assertEqual(recite(start=2, take=3), expected)
