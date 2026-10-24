import unittest

from simple_linked_list import LinkedList, EmptyListException


# No canonical data available for this exercise

class SimpleLinkedListTest(unittest.TestCase):
    def test_empty_list_has_len_zero(self):
        sut = LinkedList()
        self.assertEqual(len(sut), 0)

    def test_error_on_empty_list_head(self):
        sut = LinkedList()
        with self.assertRaises(EmptyListException) as err:
            sut.head()
        self.assertEqual(type(err.exception), EmptyListException)
        self.assertEqual(err.exception.args[0], "The list is empty.")

    def test_can_push_to_non_empty_list(self):
        sut = LinkedList([1, 2, 3])
        sut.push(4)
        self.assertEqual(len(sut), 4)

    def test_pop_from_singleton_list_removes_head(self):
        sut = LinkedList([1])
        self.assertEqual(sut.pop(), 1)
        with self.assertRaises(EmptyListException) as err:
            sut.head()
        self.assertEqual(type(err.exception), EmptyListException)
        self.assertEqual(err.exception.args[0], "The list is empty.")

    def test_singleton_list_head_has_no_next(self):
        sut = LinkedList([1])
        self.assertIsNone(sut.head().next())

    def test_singleton_linked_list_to_list_list_with_singular_element(self):
        sut = LinkedList([1])
        self.assertEqual(list(sut), [1])

    def test_reversed_singleton_list_is_same_list(self):
        sut = LinkedList([1])
        self.assertEqual(list(sut.reversed()), [1])
