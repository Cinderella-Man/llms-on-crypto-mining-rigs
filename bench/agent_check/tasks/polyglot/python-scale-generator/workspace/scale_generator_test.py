# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/scale-generator/canonical-data.json
# File last updated on 2023-07-19

import unittest

from scale_generator import (
    Scale,
)


class ScaleGeneratorTest(unittest.TestCase):

    # Test chromatic scales
    def test_chromatic_scale_with_sharps(self):
        expected = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.assertEqual(Scale("C").chromatic(), expected)

    # Test scales with specified intervals

    def test_major_scale_with_sharps(self):
        expected = ["G", "A", "B", "C", "D", "E", "F#", "G"]
        self.assertEqual(Scale("G").interval("MMmMMMm"), expected)

    def test_minor_scale_with_flats(self):
        expected = ["Bb", "C", "Db", "Eb", "F", "Gb", "Ab", "Bb"]
        self.assertEqual(Scale("bb").interval("MmMMmMM"), expected)

    def test_lydian_mode(self):
        expected = ["A", "B", "C#", "D#", "E", "F#", "G#", "A"]
        self.assertEqual(Scale("a").interval("MMMmMMm"), expected)

    def test_harmonic_minor(self):
        expected = ["D", "E", "F", "G", "A", "Bb", "Db", "D"]
        self.assertEqual(Scale("d").interval("MmMMmAm"), expected)

    def test_pentatonic(self):
        expected = ["A", "B", "C#", "E", "F#", "A"]
        self.assertEqual(Scale("A").interval("MMAMA"), expected)
