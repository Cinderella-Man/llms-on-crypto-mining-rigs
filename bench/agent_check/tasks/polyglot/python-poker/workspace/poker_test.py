# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/poker/canonical-data.json
# File last updated on 2023-12-27

import unittest

from poker import (
    best_hands,
)


class PokerTest(unittest.TestCase):
    def test_single_hand_always_wins(self):
        self.assertEqual(best_hands(["4S 5S 7H 8D JC"]), ["4S 5S 7H 8D JC"])

    def test_multiple_hands_with_the_same_high_cards_tie_compares_next_highest_ranked_down_to_last_card(
        self,
    ):
        self.assertEqual(
            best_hands(["3S 5H 6S 8D 7H", "2S 5D 6D 8C 7S"]), ["3S 5H 6S 8D 7H"]
        )

    def test_highest_pair_wins(self):
        self.assertEqual(
            best_hands(["4S 2H 6S 2D JH", "2S 4H 6C 4D JD"]), ["2S 4H 6C 4D JD"]
        )

    def test_both_hands_have_two_pairs_highest_ranked_pair_wins(self):
        self.assertEqual(
            best_hands(["2S 8H 2D 8D 3H", "4S 5H 4C 8S 5D"]), ["2S 8H 2D 8D 3H"]
        )

    def test_both_hands_have_two_pairs_that_add_to_the_same_value_win_goes_to_highest_pair(
        self,
    ):
        self.assertEqual(
            best_hands(["6S 6H 3S 3H AS", "7H 7S 2H 2S AC"]), ["7H 7S 2H 2S AC"]
        )

    def test_both_hands_have_three_of_a_kind_tie_goes_to_highest_ranked_triplet(self):
        self.assertEqual(
            best_hands(["2S 2H 2C 8D JH", "4S AH AS 8C AD"]), ["4S AH AS 8C AD"]
        )

    def test_aces_can_end_a_straight_10_j_q_k_a(self):
        self.assertEqual(
            best_hands(["4S 5H 4C 8D 4H", "10D JH QS KD AC"]), ["10D JH QS KD AC"]
        )

    def test_both_hands_with_a_straight_tie_goes_to_highest_ranked_card(self):
        self.assertEqual(
            best_hands(["4S 6C 7S 8D 5H", "5S 7H 8S 9D 6H"]), ["5S 7H 8S 9D 6H"]
        )

    def test_both_hands_have_a_flush_tie_goes_to_high_card_down_to_the_last_one_if_necessary(
        self,
    ):
        self.assertEqual(
            best_hands(["2H 7H 8H 9H 6H", "3S 5S 6S 7S 8S"]), ["2H 7H 8H 9H 6H"]
        )

    def test_with_multiple_decks_both_hands_have_a_full_house_with_the_same_triplet_tie_goes_to_the_pair(
        self,
    ):
        self.assertEqual(
            best_hands(["5H 5S 5D 9S 9D", "5H 5S 5D 8S 8D"]), ["5H 5S 5D 9S 9D"]
        )

    def test_with_multiple_decks_both_hands_with_identical_four_of_a_kind_tie_determined_by_kicker(
        self,
    ):
        self.assertEqual(
            best_hands(["3S 3H 2S 3D 3C", "3S 3H 4S 3D 3C"]), ["3S 3H 4S 3D 3C"]
        )

    def test_aces_can_start_a_straight_flush_a_2_3_4_5(self):
        self.assertEqual(
            best_hands(["KS AH AS AD AC", "4H AH 3H 2H 5H"]), ["4H AH 3H 2H 5H"]
        )

    def test_even_though_an_ace_is_usually_high_a_5_high_straight_flush_is_the_lowest_scoring_straight_flush(
        self,
    ):
        self.assertEqual(
            best_hands(["2H 3H 4H 5H 6H", "4D AD 3D 2D 5D"]), ["2H 3H 4H 5H 6H"]
        )
