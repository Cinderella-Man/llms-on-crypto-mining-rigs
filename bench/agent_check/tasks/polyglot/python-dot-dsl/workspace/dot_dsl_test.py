import unittest

from dot_dsl import Graph, Node, Edge, NODE, EDGE, ATTR


class DotDslTest(unittest.TestCase):
    def test_empty_graph(self):
        g = Graph()

        self.assertEqual(g.nodes, [])
        self.assertEqual(g.edges, [])
        self.assertEqual(g.attrs, {})

    def test_graph_with_one_edge(self):
        g = Graph([
            (EDGE, "a", "b", {})
        ])

        self.assertEqual(g.nodes, [])
        self.assertEqual(g.edges, [Edge("a", "b", {})])
        self.assertEqual(g.attrs, {})

    def test_malformed_graph(self):
        with self.assertRaises(TypeError) as err:
            Graph(1)
        self.assertEqual(type(err.exception), TypeError)
        self.assertEqual(err.exception.args[0], "Graph data malformed")

        with self.assertRaises(TypeError) as err:
            Graph("problematic")
        self.assertEqual(type(err.exception), TypeError)
        self.assertEqual(err.exception.args[0], "Graph data malformed")


    def test_malformed_node(self):
        with self.assertRaises(ValueError) as err:
            Graph([(NODE, 1, 2, 3)])
        self.assertEqual(type(err.exception), ValueError)
        self.assertEqual(err.exception.args[0], "Node is malformed")
