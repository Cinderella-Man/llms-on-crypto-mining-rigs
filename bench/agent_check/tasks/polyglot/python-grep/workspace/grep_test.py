# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/grep/canonical-data.json
# File last updated on 2023-07-19

import io
import unittest

from grep import (
    grep,
)
from unittest import mock

FILE_TEXT = {
    "iliad.txt": """Achilles sing, O Goddess! Peleus' son;
His wrath pernicious, who ten thousand woes
Caused to Achaia's host, sent many a soul
Illustrious into Ades premature,
And Heroes gave (so stood the will of Jove)
To dogs and to all ravening fowls a prey,
When fierce dispute had separated once
The noble Chief Achilles from the son
Of Atreus, Agamemnon, King of men.\n""",
    "midsummer-night.txt": """I do entreat your grace to pardon me.
I know not by what power I am made bold,
Nor how it may concern my modesty,
In such a presence here to plead my thoughts;
But I beseech your grace that I may know
The worst that may befall me in this case,
If I refuse to wed Demetrius.\n""",
    "paradise-lost.txt": """Of Mans First Disobedience, and the Fruit
Of that Forbidden Tree, whose mortal tast
Brought Death into the World, and all our woe,
With loss of Eden, till one greater Man
Restore us, and regain the blissful Seat,
Sing Heav'nly Muse, that on the secret top
Of Oreb, or of Sinai, didst inspire
That Shepherd, who first taught the chosen Seed\n""",
}


def open_mock(fname, *args, **kwargs):
    try:
        return io.StringIO(FILE_TEXT[fname])
    except KeyError:
        raise RuntimeError(
            "Expected one of {0!r}: got {1!r}".format(list(FILE_TEXT.keys()), fname)
        )


@mock.patch("grep.open", name="open", side_effect=open_mock, create=True)
@mock.patch("io.StringIO", name="StringIO", wraps=io.StringIO)
class GrepTest(unittest.TestCase):
    # Test grepping a single file
    def test_one_file_one_match_no_flags(self, mock_file, mock_open):
        self.assertMultiLineEqual(
            grep("Agamemnon", "", ["iliad.txt"]), "Of Atreus, Agamemnon, King of men.\n"
        )

    def test_one_file_one_match_print_file_names_flag(self, mock_file, mock_open):
        self.assertMultiLineEqual(
            grep("Forbidden", "-l", ["paradise-lost.txt"]), "paradise-lost.txt\n"
        )

    def test_one_file_several_matches_no_flags(self, mock_file, mock_open):
        self.assertMultiLineEqual(
            grep("may", "", ["midsummer-night.txt"]),
            "Nor how it may concern my modesty,\n"
            "But I beseech your grace that I may know\n"
            "The worst that may befall me in this case,\n",
        )

    def test_one_file_several_matches_case_insensitive_flag(self, mock_file, mock_open):
        self.assertMultiLineEqual(
            grep("ACHILLES", "-i", ["iliad.txt"]),
            "Achilles sing, O Goddess! Peleus' son;\n"
            "The noble Chief Achilles from the son\n",
        )

    def test_one_file_one_match_file_flag_takes_precedence_over_line_flag(
        self, mock_file, mock_open
    ):
        self.assertMultiLineEqual(grep("ten", "-n -l", ["iliad.txt"]), "iliad.txt\n")

    # Test grepping multiples files at once

    def test_multiple_files_several_matches_no_flags(self, mock_file, mock_open):
        self.assertMultiLineEqual(
            grep("may", "", ["iliad.txt", "midsummer-night.txt", "paradise-lost.txt"]),
            "midsummer-night.txt:Nor how it may concern my modesty,\n"
            "midsummer-night.txt:But I beseech your grace that I may know\n"
            "midsummer-night.txt:The worst that may befall me in this case,\n",
        )

    def test_multiple_files_several_matches_case_insensitive_flag(
        self, mock_file, mock_open
    ):
        self.assertMultiLineEqual(
            grep("TO", "-i", ["iliad.txt", "midsummer-night.txt", "paradise-lost.txt"]),
            "iliad.txt:Caused to Achaia's host, sent many a soul\n"
            "iliad.txt:Illustrious into Ades premature,\n"
            "iliad.txt:And Heroes gave (so stood the will of Jove)\n"
            "iliad.txt:To dogs and to all ravening fowls a prey,\n"
            "midsummer-night.txt:I do entreat your grace to pardon me.\n"
            "midsummer-night.txt:In such a presence here to plead my thoughts;\n"
            "midsummer-night.txt:If I refuse to wed Demetrius.\n"
            "paradise-lost.txt:Brought Death into the World, and all our woe,\n"
            "paradise-lost.txt:Restore us, and regain the blissful Seat,\n"
            "paradise-lost.txt:Sing Heav'nly Muse, that on the secret top\n",
        )

    def test_multiple_files_one_match_multiple_flags(self, mock_file, mock_open):
        self.assertMultiLineEqual(
            grep(
                "WITH LOSS OF EDEN, TILL ONE GREATER MAN",
                "-n -i -x",
                ["iliad.txt", "midsummer-night.txt", "paradise-lost.txt"],
            ),
            "paradise-lost.txt:4:With loss of Eden, till one greater Man\n",
        )

    def test_multiple_files_several_matches_inverted_and_match_entire_lines_flags(
        self, mock_file, mock_open
    ):
        self.assertMultiLineEqual(
            grep(
                "Illustrious into Ades premature,",
                "-x -v",
                ["iliad.txt", "midsummer-night.txt", "paradise-lost.txt"],
            ),
            "iliad.txt:Achilles sing, O Goddess! Peleus' son;\n"
            "iliad.txt:His wrath pernicious, who ten thousand woes\n"
            "iliad.txt:Caused to Achaia's host, sent many a soul\n"
            "iliad.txt:And Heroes gave (so stood the will of Jove)\n"
            "iliad.txt:To dogs and to all ravening fowls a prey,\n"
            "iliad.txt:When fierce dispute had separated once\n"
            "iliad.txt:The noble Chief Achilles from the son\n"
            "iliad.txt:Of Atreus, Agamemnon, King of men.\n"
            "midsummer-night.txt:I do entreat your grace to pardon me.\n"
            "midsummer-night.txt:I know not by what power I am made bold,\n"
            "midsummer-night.txt:Nor how it may concern my modesty,\n"
            "midsummer-night.txt:In such a presence here to plead my thoughts;\n"
            "midsummer-night.txt:But I beseech your grace that I may know\n"
            "midsummer-night.txt:The worst that may befall me in this case,\n"
            "midsummer-night.txt:If I refuse to wed Demetrius.\n"
            "paradise-lost.txt:Of Mans First Disobedience, and the Fruit\n"
            "paradise-lost.txt:Of that Forbidden Tree, whose mortal tast\n"
            "paradise-lost.txt:Brought Death into the World, and all our woe,\n"
            "paradise-lost.txt:With loss of Eden, till one greater Man\n"
            "paradise-lost.txt:Restore us, and regain the blissful Seat,\n"
            "paradise-lost.txt:Sing Heav'nly Muse, that on the secret top\n"
            "paradise-lost.txt:Of Oreb, or of Sinai, didst inspire\n"
            "paradise-lost.txt:That Shepherd, who first taught the chosen Seed\n",
        )
