# These tests are auto-generated with test data from:
# https://github.com/exercism/problem-specifications/tree/main/exercises/react/canonical-data.json
# File last updated on 2023-07-19

from functools import partial
import unittest

from react import (
    InputCell,
    ComputeCell,
)


class ReactTest(unittest.TestCase):
    def test_input_cells_have_a_value(self):
        input = InputCell(10)
        self.assertEqual(input.value, 10)

    def test_compute_cells_take_inputs_in_the_right_order(self):
        one = InputCell(1)
        two = InputCell(2)
        output = ComputeCell(
            [
                one,
                two,
            ],
            lambda inputs: inputs[0] + inputs[1] * 10,
        )
        self.assertEqual(output.value, 21)

    def test_compute_cells_fire_callbacks(self):
        input = InputCell(1)
        output = ComputeCell(
            [
                input,
            ],
            lambda inputs: inputs[0] + 1,
        )
        cb1_observer = []
        callback1 = self.callback_factory(cb1_observer)
        output.add_callback(callback1)
        input.value = 3
        self.assertEqual(cb1_observer[-1], 4)

    def test_callbacks_can_fire_from_multiple_cells(self):
        input = InputCell(1)
        plus_one = ComputeCell(
            [
                input,
            ],
            lambda inputs: inputs[0] + 1,
        )
        minus_one = ComputeCell(
            [
                input,
            ],
            lambda inputs: inputs[0] - 1,
        )
        cb1_observer = []
        cb2_observer = []
        callback1 = self.callback_factory(cb1_observer)
        callback2 = self.callback_factory(cb2_observer)
        plus_one.add_callback(callback1)
        minus_one.add_callback(callback2)
        input.value = 10
        self.assertEqual(cb1_observer[-1], 11)
        self.assertEqual(cb2_observer[-1], 9)

    def test_callbacks_should_only_be_called_once_even_if_multiple_dependencies_change(
        self,
    ):
        input = InputCell(1)
        plus_one = ComputeCell(
            [
                input,
            ],
            lambda inputs: inputs[0] + 1,
        )
        minus_one1 = ComputeCell(
            [
                input,
            ],
            lambda inputs: inputs[0] - 1,
        )
        minus_one2 = ComputeCell(
            [
                minus_one1,
            ],
            lambda inputs: inputs[0] - 1,
        )
        output = ComputeCell(
            [
                plus_one,
                minus_one2,
            ],
            lambda inputs: inputs[0] * inputs[1],
        )
        cb1_observer = []
        callback1 = self.callback_factory(cb1_observer)
        output.add_callback(callback1)
        input.value = 4
        self.assertEqual(cb1_observer[-1], 10)

    # Utility functions.
    def callback_factory(self, observer):
        def callback(observer, value):
            observer.append(value)

        return partial(callback, observer)
