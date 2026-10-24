import unittest

import hangman
from hangman import Hangman


# Tests adapted from csharp//hangman/HangmanTest.cs

class HangmanTests(unittest.TestCase):
    def test_initially_9_failures_are_allowed(self):
        game = Hangman('foo')
        self.assertEqual(game.get_status(), hangman.STATUS_ONGOING)
        self.assertEqual(game.remaining_guesses, 9)

    def test_feeding_a_correct_letter_removes_underscores(self):
        game = Hangman('foobar')

        game.guess('b')
        self.assertEqual(game.get_status(), hangman.STATUS_ONGOING)
        self.assertEqual(game.remaining_guesses, 9)
        self.assertEqual(game.get_masked_word(), '___b__')

        game.guess('o')
        self.assertEqual(game.get_status(), hangman.STATUS_ONGOING)
        self.assertEqual(game.remaining_guesses, 9)
        self.assertEqual(game.get_masked_word(), '_oob__')

    def test_winning_on_last_guess_still_counts_as_a_win(self):
        game = Hangman('aaa')
        for ch in 'bcdefghij':
            game.guess(ch)
        game.guess('a')
        self.assertEqual(game.remaining_guesses, 0)
        self.assertEqual(game.get_status(), hangman.STATUS_WIN)
        self.assertEqual(game.get_masked_word(), 'aaa')
