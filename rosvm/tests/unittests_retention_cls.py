####
#
# The MIT License (MIT)
#
# Copyright 2017-2019 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

import unittest
import numpy as np

from collections import OrderedDict

from ranksvm.retentiongraph_cls import RetentionGraph


class Test_load_data_from_target(unittest.TestCase):
    def test_simplecases(self):
        cretention = RetentionGraph()

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2,
                    ("M1", "B"): 2, ("M2", "B"): 1}

        cretention.load_data_from_target(d_target)

        self.assertEqual(len(cretention.lrows), 2)
        self.assertIn(["M1", "M2", 1, "A"], cretention.lrows)
        self.assertIn(["M2", "M1", 1, "B"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target(d_target)

        self.assertEqual(len(cretention.lrows), 2)
        self.assertIn(["M1", "M2", 1, "A"], cretention.lrows)
        self.assertIn(["M1", "M2", 1, "B"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2,  ("M3", "A"): 3, ("M4", "A"): 4,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target(d_target)

        self.assertEqual(len(cretention.lrows), 4)
        self.assertIn(["M1", "M2", 1, "A"], cretention.lrows)
        self.assertIn(["M2", "M3", 1, "A"], cretention.lrows)
        self.assertIn(["M3", "M4", 1, "A"], cretention.lrows)
        self.assertIn(["M1", "M2", 1, "B"], cretention.lrows)

    def test_bordercases(self):
        cretention = RetentionGraph()

        # ----------------------------------------------------
        d_target = {}

        cretention.load_data_from_target(d_target)

        self.assertEqual(cretention.lrows, [])

        # ----------------------------------------------------
        d_target = {("M2", "B"): 7}

        cretention.load_data_from_target(d_target)

        self.assertEqual(cretention.lrows, [])

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "B"): 7}

        cretention.load_data_from_target(d_target)

        self.assertEqual(cretention.lrows, [])

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 3, ("M2", "B"): 7, ("M3", "B"): 19}

        cretention.load_data_from_target(d_target, linclude_collection=[])

        self.assertEqual(cretention.lrows, [])

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 3, ("M2", "B"): 7, ("M3", "B"): 19}

        cretention.load_data_from_target(d_target, linclude_node=[])

        self.assertEqual(cretention.lrows, [])

    def test_complexcases(self):
        """
        TEST: Nodes with the same target value are handled correctly.
        """
        cretention = RetentionGraph()

        # ----------------------------------------------------
        d_target = {("M1", "A"): 7, ("M2", "A"): 7, ("M3", "A"): 3, ("M4", "A"): 8,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target(d_target)

        self.assertEqual(len(cretention.lrows), 7)
        self.assertIn(["M3", "M1", 1, "A"], cretention.lrows)
        self.assertIn(["M3", "M2", 1, "A"], cretention.lrows)
        self.assertIn(["M1", "M2", 0, "A"], cretention.lrows)
        self.assertIn(["M2", "M1", 0, "A"], cretention.lrows)
        self.assertIn(["M1", "M4", 1, "A"], cretention.lrows)
        self.assertIn(["M2", "M4", 1, "A"], cretention.lrows)
        self.assertIn(["M1", "M2", 1, "B"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 7, ("M2", "A"): 7, ("M3", "A"): 7}

        cretention.load_data_from_target(d_target)

        self.assertEqual(len(cretention.lrows), 6)
        self.assertIn(["M1", "M2", 0, "A"], cretention.lrows)
        self.assertIn(["M1", "M3", 0, "A"], cretention.lrows)
        self.assertIn(["M2", "M1", 0, "A"], cretention.lrows)
        self.assertIn(["M2", "M3", 0, "A"], cretention.lrows)
        self.assertIn(["M3", "M1", 0, "A"], cretention.lrows)
        self.assertIn(["M3", "M2", 0, "A"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 7, ("M2", "A"): 7, ("M3", "A"): 7, ("M4", "A"): 1}

        cretention.load_data_from_target(d_target)

        self.assertEqual(len(cretention.lrows), 9)
        self.assertIn(["M4", "M1", 1, "A"], cretention.lrows)
        self.assertIn(["M4", "M2", 1, "A"], cretention.lrows)
        self.assertIn(["M4", "M3", 1, "A"], cretention.lrows)
        self.assertIn(["M1", "M2", 0, "A"], cretention.lrows)
        self.assertIn(["M1", "M3", 0, "A"], cretention.lrows)
        self.assertIn(["M2", "M1", 0, "A"], cretention.lrows)
        self.assertIn(["M2", "M3", 0, "A"], cretention.lrows)
        self.assertIn(["M3", "M1", 0, "A"], cretention.lrows)
        self.assertIn(["M3", "M2", 0, "A"], cretention.lrows)

    def test_exclude_nodes_and_collections(self):
        """
        TEST: Nodes and collections are ignored if specified.
        """
        cretention = RetentionGraph()

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2, ("M3", "A"): 3, ("M4", "A"): 4,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target(d_target, linclude_collection="B")

        self.assertEqual(len(cretention.lrows), 1)
        self.assertIn(["M1", "M2", 1, "B"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2, ("M3", "A"): 3, ("M4", "A"): 4,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target(d_target, linclude_collection="A")

        self.assertEqual(len(cretention.lrows), 3)
        self.assertIn(["M1", "M2", 1, "A"], cretention.lrows)
        self.assertIn(["M2", "M3", 1, "A"], cretention.lrows)
        self.assertIn(["M3", "M4", 1, "A"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 1, ("M2", "A"): 2, ("M3", "A"): 3, ("M4", "A"): 4,
                    ("M1", "B"): 1, ("M2", "B"): 2}

        cretention.load_data_from_target(d_target, linclude_collection="A", linclude_node=["M1", "M4"])

        self.assertEqual(len(cretention.lrows), 1)
        self.assertIn(["M1", "M4", 1, "A"], cretention.lrows)

        # ----------------------------------------------------
        d_target = {("M1", "A"): 7, ("M2", "A"): 7, ("M3", "A"): 7, ("M4", "A"): 1}

        cretention.load_data_from_target(d_target, linclude_node=["M1", "M2", "M4"])

        self.assertEqual(len(cretention.lrows), 4)
        self.assertIn(["M4", "M1", 1, "A"], cretention.lrows)
        self.assertIn(["M4", "M2", 1, "A"], cretention.lrows)
        self.assertIn(["M1", "M2", 0, "A"], cretention.lrows)
        self.assertIn(["M2", "M1", 0, "A"], cretention.lrows)


class Test_upper_lower_set_node(unittest.TestCase):
    def test_simplecases(self):
        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("M1", "A"): 1, ("M2", "A"): 2,  ("M3", "A"): 3, ("M4", "A"): 4,
                    ("M1", "B"): 1, ("M5", "B"): 2}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_node(cretention.dG)
        dm, dc = cretention.dmolecules, cretention.dcollections

        self.assertEqual(len(d_moleculecut), len(d_target))
        # Check distances
        self.assertEqual(d_moleculecut[(dm["M1"], dc["A"])][0], {(dm["M1"], dc["B"]): 0})
        self.assertEqual(d_moleculecut[(dm["M1"], dc["B"])][0], {(dm["M1"], dc["A"]): 0})
        self.assertEqual(d_moleculecut[(dm["M1"], dc["A"])][1],
                         {(dm["M1"], dc["B"]): 0, (dm["M2"], dc["A"]): 1,
                          (dm["M3"], dc["A"]): 2, (dm["M4"], dc["A"]): 3,
                          (dm["M5"], dc["B"]): 1})
        self.assertEqual(d_moleculecut[(dm["M2"], dc["A"])][1],
                         {(dm["M3"], dc["A"]): 1, (dm["M4"], dc["A"]): 2})
        self.assertEqual(d_moleculecut[(dm["M2"], dc["A"])][0],
                         {(dm["M1"], dc["A"]): 1, (dm["M1"], dc["B"]): 1})

        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 3,
                    ("E", "2"): 1, ("B", "2"): 2, ("C", "2"): 3, ("D", "2"): 4}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_node(cretention.dG, str_as_idf=True)

        self.assertEqual(len(d_moleculecut), len(d_target))

        # Check Upper [0] and Lower [1] sets
        self.assertEqual(d_moleculecut[("A", "1")][0], dict())
        self.assertEqual(d_moleculecut[("A", "1")][1],
                         dict([(("B", "1"), 1), (("B", "2"), 1), (("C", "1"), 2), (("C", "2"), 2), (("D", "2"), 3)]))

        self.assertEqual(d_moleculecut[("B", "1")][0],
                         dict([(("E", "2"), 1), (("A", "1"), 1), (("B", "2"), 0)]))
        self.assertEqual(d_moleculecut[("B", "1")][1],
                         dict([(("B", "2"), 0), (("C", "1"), 1), (("C", "2"), 1), (("D", "2"), 2)]))

        self.assertEqual(d_moleculecut[("B", "2")][0],
                         dict([(("E", "2"), 1), (("A", "1"), 1), (("B", "1"), 0)]))
        self.assertEqual(d_moleculecut[("B", "2")][1],
                         dict([(("B", "1"), 0), (("C", "1"), 1), (("C", "2"), 1), (("D", "2"), 2)]))

        self.assertEqual(d_moleculecut[("C", "1")][0],
                         dict([(("B", "1"), 1), (("B", "2"), 1), (("C", "2"), 0), (("A", "1"), 2), (("E", "2"), 2)]))
        self.assertEqual(d_moleculecut[("C", "1")][1],
                         dict([(("C", "2"), 0), (("D", "2"), 1)]))

        self.assertEqual(d_moleculecut[("C", "2")][0],
                         dict([(("B", "1"), 1), (("B", "2"), 1), (("C", "1"), 0), (("A", "1"), 2), (("E", "2"), 2)]))
        self.assertEqual(d_moleculecut[("C", "2")][1],
                         dict([(("C", "1"), 0), (("D", "2"), 1)]))

        self.assertEqual(d_moleculecut[("D", "2")][0],
                         dict([(("B", "1"), 2), (("B", "2"), 2), (("C", "1"), 1), (("C", "2"), 1), (("A", "1"), 3),
                               (("E", "2"), 3)]))
        self.assertEqual(d_moleculecut[("D", "2")][1], dict())

        self.assertEqual(d_moleculecut[("E", "2")][0], dict())
        self.assertEqual(d_moleculecut[("E", "2")][1],
                         dict([(("B", "1"), 1), (("B", "2"), 1), (("C", "1"), 2), (("C", "2"), 2), (("D", "2"), 3)]))


class Test_upper_lower_set_molecule(unittest.TestCase):
    def test_simplecases(self):
        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 3,
                    ("E", "2"): 1, ("B", "2"): 2, ("C", "2"): 3, ("D", "2"): 4}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_molecule(cretention.dG, str_as_idf=True)

        self.assertEqual(len(d_moleculecut), 5)  # number of unique molecules

        # Check Upper [0] and Lower [1] sets
        self.assertEqual(d_moleculecut["A"][0], dict())
        self.assertEqual(d_moleculecut["A"][1], dict([("B", 1), ("C", 2), ("D", 3)]))
        self.assertEqual(d_moleculecut["B"][0], dict([("A", 1), ("E", 1)]))
        self.assertEqual(d_moleculecut["B"][1], dict([("C", 1), ("D", 2)]))
        self.assertEqual(d_moleculecut["C"][0], dict([("A", 2), ("E", 2), ("B", 1)]))
        self.assertEqual(d_moleculecut["C"][1], dict([("D", 1)]))
        self.assertEqual(d_moleculecut["D"][0], dict([("C", 1), ("B", 2), ("A", 3), ("E", 3)]))
        self.assertEqual(d_moleculecut["D"][1], dict())
        self.assertEqual(d_moleculecut["E"][0], dict())
        self.assertEqual(d_moleculecut["E"][1], dict([("B", 1), ("C", 2), ("D", 3)]))

        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 3,
                    ("E", "2"): 1, ("B", "2"): 2, ("C", "2"): 3, ("D", "2"): 4}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph(ireverse=False)  # no inter-dataset connection
        d_moleculecut = cretention.upper_lower_set_molecule(cretention.dG, str_as_idf=True)

        self.assertEqual(len(d_moleculecut), 5)  # number of unique molecules

        # Check Upper [0] and Lower [1] sets
        self.assertEqual(d_moleculecut["A"][0], dict())
        self.assertEqual(d_moleculecut["A"][1], dict([("B", 1), ("C", 2)]))
        self.assertEqual(d_moleculecut["B"][0], dict([("A", 1), ("E", 1)]))
        self.assertEqual(d_moleculecut["B"][1], dict([("C", 1), ("D", 2)]))
        self.assertEqual(d_moleculecut["C"][0], dict([("A", 2), ("E", 2), ("B", 1)]))
        self.assertEqual(d_moleculecut["C"][1], dict([("D", 1)]))
        self.assertEqual(d_moleculecut["D"][0], dict([("C", 1), ("B", 2), ("E", 3)]))
        self.assertEqual(d_moleculecut["D"][1], dict())
        self.assertEqual(d_moleculecut["E"][0], dict())
        self.assertEqual(d_moleculecut["E"][1], dict([("B", 1), ("C", 2), ("D", 3)]))

    def test_more_complex_cases(self):
        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 3, ("D", "1"): 9, ("E", "1"): 11,
                    ("F", "2"): 4, ("D", "2"): 6, ("C", "2"): 7,
                    ("S", "3"): 4, ("T", "3"): 5, ("U", "3"): 9}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_molecule(cretention.dG, str_as_idf=True)

        self.assertEqual(len(d_moleculecut), 9)  # number of unique molecules

        # Check Upper [0] and Lower [1] sets
        self.assertEqual(d_moleculecut["A"][0], dict())
        self.assertEqual(d_moleculecut["A"][1], dict([("B", 1), ("C", 2), ("D", 3), ("E", 4)]))

        self.assertEqual(d_moleculecut["B"][0], dict([("A", 1)]))
        self.assertEqual(d_moleculecut["B"][1], dict([("C", 1), ("D", 2), ("E", 3)]))

        self.assertEqual(d_moleculecut["C"][0], dict([("D", 1), ("B", 1), ("A", 2), ("F", 2)]))
        self.assertEqual(d_moleculecut["C"][1], dict([("D", 1), ("E", 2)]))

        self.assertEqual(d_moleculecut["D"][0], dict([("C", 1), ("B", 2), ("A", 3), ("F", 1)]))
        self.assertEqual(d_moleculecut["D"][1], dict([("C", 1), ("E", 1)]))

        self.assertEqual(d_moleculecut["E"][0], dict([("D", 1), ("C", 2), ("F", 2), ("B", 3), ("A", 4)]))
        self.assertEqual(d_moleculecut["E"][1], dict())

        self.assertEqual(d_moleculecut["F"][0], dict())
        self.assertEqual(d_moleculecut["F"][1], dict([("D", 1), ("C", 2), ("E", 2)]))

        self.assertEqual(d_moleculecut["S"][0], dict())
        self.assertEqual(d_moleculecut["S"][1], dict([("T", 1), ("U", 2)]))

        self.assertEqual(d_moleculecut["T"][0], dict([("S", 1)]))
        self.assertEqual(d_moleculecut["T"][1], dict([("U", 1)]))

        self.assertEqual(d_moleculecut["U"][0], dict([("T", 1), ("S", 2)]))
        self.assertEqual(d_moleculecut["U"][1], dict())

        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 2, ("D", "1"): 2, ("E", "1"): 11,
                    ("F", "2"): 4, ("D", "2"): 6, ("G", "2"): 7}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_molecule(cretention.dG, str_as_idf=True)

        self.assertEqual(len(d_moleculecut), 7)  # number of unique molecules

        # Check Upper [0] and Lower [1] sets
        self.assertEqual(d_moleculecut["A"][0], dict())
        self.assertEqual(d_moleculecut["A"][1], dict([("B", 1), ("C", 1), ("D", 1), ("E", 2), ("G", 2)]))

        self.assertEqual(d_moleculecut["B"][0], dict([("A", 1)]))
        self.assertEqual(d_moleculecut["B"][1], dict([("E", 1)]))

        self.assertEqual(d_moleculecut["C"][0], dict([("A", 1)]))
        self.assertEqual(d_moleculecut["C"][1], dict([("E", 1)]))

        self.assertEqual(d_moleculecut["D"][0], dict([("A", 1), ("F", 1)]))
        self.assertEqual(d_moleculecut["D"][1], dict([("E", 1), ("G", 1)]))

        self.assertEqual(d_moleculecut["E"][0], dict([("B", 1), ("C", 1), ("D", 1), ("A", 2), ("F", 2)]))
        self.assertEqual(d_moleculecut["E"][1], dict())

        self.assertEqual(d_moleculecut["F"][0], dict())
        self.assertEqual(d_moleculecut["F"][1], dict([("D", 1), ("E", 2), ("G", 2)]))

        self.assertEqual(d_moleculecut["G"][0], dict([("D", 1), ("A", 2), ("F", 2)]))
        self.assertEqual(d_moleculecut["G"][1], dict())


class Test_Order_Tanimoto_Kernel(unittest.TestCase):
    def test_upper_overlap(self):
        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 3,
                    ("E", "2"): 1, ("B", "2"): 2, ("C", "2"): 3, ("D", "2"): 4}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_molecule(cretention.dG, str_as_idf=True)

        dm, dc = cretention.dmolecules, cretention.dcollections
        self.assertIsInstance(dm, OrderedDict)
        self.assertIsInstance(dc, OrderedDict)

        K_us, n2r_us = RetentionGraph.order_tanimoto_kernel(d_moleculecut, type="upper", overlap=True)
        self.assertEqual(K_us.shape, (5, 5))
        np.testing.assert_equal(np.diag(K_us), np.ones(5))
        np.testing.assert_equal(K_us, K_us.T)
        self.assertTrue(np.all(K_us <= 1.0))
        self.assertTrue(np.all(K_us >= 0.0))

        # self.assertEqual(d_moleculecut["A"][0], dict())
        # self.assertEqual(d_moleculecut["B"][0], dict([("A", 1), ("E", 1)]))
        # self.assertEqual(d_moleculecut["C"][0], dict([("A", 2), ("E", 2), ("B", 1)]))
        # self.assertEqual(d_moleculecut["D"][0], dict([("C", 1), ("B", 2), ("A", 3), ("E", 3)]))
        # self.assertEqual(d_moleculecut["E"][0], dict())
        self.assertEqual(K_us[n2r_us["A"], n2r_us["B"]], 0.0)
        self.assertEqual(K_us[n2r_us["A"], n2r_us["C"]], 0.0)
        self.assertEqual(K_us[n2r_us["B"], n2r_us["C"]], 2. / 3.)
        self.assertEqual(K_us[n2r_us["B"], n2r_us["D"]], 2. / 4.)
        self.assertEqual(K_us[n2r_us["B"], n2r_us["E"]], 0.0)
        self.assertEqual(K_us[n2r_us["C"], n2r_us["D"]], 3. / 4.)
        self.assertEqual(K_us[n2r_us["C"], n2r_us["E"]], 0.0)

        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 3, ("D", "1"): 9, ("E", "1"): 11,
                    ("F", "2"): 4, ("D", "2"): 6, ("C", "2"): 7,
                    ("S", "3"): 4, ("T", "3"): 5, ("U", "3"): 9}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_molecule(cretention.dG, str_as_idf=True)

        dm, dc = cretention.dmolecules, cretention.dcollections
        self.assertIsInstance(dm, OrderedDict)
        self.assertIsInstance(dc, OrderedDict)

        K_us, n2r_us = RetentionGraph.order_tanimoto_kernel(d_moleculecut, type="upper", overlap=True)
        self.assertEqual(K_us.shape, (9, 9))
        np.testing.assert_equal(np.diag(K_us), np.ones(9))
        np.testing.assert_equal(K_us, K_us.T)
        self.assertTrue(np.all(K_us <= 1.0))
        self.assertTrue(np.all(K_us >= 0.0))

        # self.assertEqual(d_moleculecut["A"][0], dict())
        # self.assertEqual(d_moleculecut["B"][0], dict([("A", 1)]))
        # self.assertEqual(d_moleculecut["C"][0], dict([("D", 1), ("B", 1), ("A", 2), ("F", 2)]))
        # self.assertEqual(d_moleculecut["D"][0], dict([("C", 1), ("B", 2), ("A", 3), ("F", 1)]))
        # self.assertEqual(d_moleculecut["E"][0], dict([("D", 1), ("C", 2), ("F", 2), ("B", 3), ("A", 4)]))
        # self.assertEqual(d_moleculecut["F"][0], dict())
        # self.assertEqual(d_moleculecut["S"][0], dict())
        # self.assertEqual(d_moleculecut["T"][0], dict([("S", 1)]))
        # self.assertEqual(d_moleculecut["U"][0], dict([("T", 1), ("S", 2)]))
        self.assertEqual(K_us[n2r_us["A"], n2r_us["B"]], 0.0)
        self.assertEqual(K_us[n2r_us["A"], n2r_us["C"]], 0.0)
        self.assertEqual(K_us[n2r_us["B"], n2r_us["C"]], 1. / 4.)
        self.assertEqual(K_us[n2r_us["B"], n2r_us["D"]], 1. / 4.)
        self.assertEqual(K_us[n2r_us["B"], n2r_us["E"]], 1. / 5.)
        self.assertEqual(K_us[n2r_us["B"], n2r_us["T"]], 0.0)
        self.assertEqual(K_us[n2r_us["C"], n2r_us["D"]], 3. / 5.)
        self.assertEqual(K_us[n2r_us["C"], n2r_us["E"]], 4. / 5.)
        self.assertEqual(K_us[n2r_us["C"], n2r_us["U"]], 0.0)
        self.assertEqual(K_us[n2r_us["D"], n2r_us["E"]], 4. / 5.)
        self.assertEqual(K_us[n2r_us["T"], n2r_us["U"]], 1. / 2.)

    def test_upper_nooverlap(self):
        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 3, ("D", "1"): 9, ("E", "1"): 11,
                    ("F", "2"): 4, ("D", "2"): 6, ("C", "2"): 7,
                    ("S", "3"): 4, ("T", "3"): 5, ("U", "3"): 9}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_molecule(cretention.dG, str_as_idf=True)

        dm, dc = cretention.dmolecules, cretention.dcollections
        self.assertIsInstance(dm, OrderedDict)
        self.assertIsInstance(dc, OrderedDict)

        K_us, n2r_us = RetentionGraph.order_tanimoto_kernel(d_moleculecut, type="upper", overlap=False)
        self.assertEqual(K_us.shape, (9, 9))
        np.testing.assert_equal(np.diag(K_us), np.ones(9))
        np.testing.assert_equal(K_us, K_us.T)
        self.assertTrue(np.all(K_us <= 1.0))
        self.assertTrue(np.all(K_us >= 0.0))

        # self.assertEqual(d_moleculecut["A"][0], dict())
        # self.assertEqual(d_moleculecut["B"][0], dict([("A", 1)]))
        # self.assertEqual(d_moleculecut["C"][0], dict([("B", 1), ("A", 2), ("F", 2)]))
        # self.assertEqual(d_moleculecut["D"][0], dict([("B", 2), ("A", 3), ("F", 1)]))
        # self.assertEqual(d_moleculecut["E"][0], dict([("D", 1), ("C", 2), ("F", 2), ("B", 3), ("A", 4)]))
        # self.assertEqual(d_moleculecut["F"][0], dict())
        # self.assertEqual(d_moleculecut["S"][0], dict())
        # self.assertEqual(d_moleculecut["T"][0], dict([("S", 1)]))
        # self.assertEqual(d_moleculecut["U"][0], dict([("T", 1), ("S", 2)]))
        self.assertEqual(K_us[n2r_us["C"], n2r_us["D"]], 1.)
        self.assertEqual(K_us[n2r_us["C"], n2r_us["E"]], 3. / 5.)
        self.assertEqual(K_us[n2r_us["C"], n2r_us["U"]], 0.0)
        self.assertEqual(K_us[n2r_us["D"], n2r_us["E"]], 3. / 5.)

    def test_lower_nooverlap(self):
        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 3, ("D", "1"): 9, ("E", "1"): 11,
                    ("F", "2"): 4, ("D", "2"): 6, ("C", "2"): 7,
                    ("S", "3"): 4, ("T", "3"): 5, ("U", "3"): 9}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_molecule(cretention.dG, str_as_idf=True)

        dm, dc = cretention.dmolecules, cretention.dcollections
        self.assertIsInstance(dm, OrderedDict)
        self.assertIsInstance(dc, OrderedDict)

        K_us, n2r_us = RetentionGraph.order_tanimoto_kernel(d_moleculecut, type="lower", overlap=False)
        self.assertEqual(K_us.shape, (9, 9))
        np.testing.assert_equal(np.diag(K_us), np.ones(9))
        np.testing.assert_equal(K_us, K_us.T)
        self.assertTrue(np.all(K_us <= 1.0))
        self.assertTrue(np.all(K_us >= 0.0))

        # self.assertEqual(d_moleculecut["A"][1], dict([("B", 1), ("C", 2), ("D", 3), ("E", 4)]))
        # self.assertEqual(d_moleculecut["B"][1], dict([("C", 1), ("D", 2), ("E", 3)]))
        # self.assertEqual(d_moleculecut["C"][1], dict([("E", 2)]))
        # self.assertEqual(d_moleculecut["D"][1], dict([("E", 1)]))
        # self.assertEqual(d_moleculecut["E"][1], dict())
        # self.assertEqual(d_moleculecut["F"][1], dict([("D", 1), ("C", 2), ("E", 2)]))
        # self.assertEqual(d_moleculecut["S"][1], dict([("T", 1), ("U", 2)]))
        # self.assertEqual(d_moleculecut["T"][1], dict([("U", 1)]))
        # self.assertEqual(d_moleculecut["U"][1], dict())
        self.assertEqual(K_us[n2r_us["C"], n2r_us["D"]], 1.)
        self.assertEqual(K_us[n2r_us["C"], n2r_us["E"]], 0.0)
        self.assertEqual(K_us[n2r_us["C"], n2r_us["U"]], 0.0)
        self.assertEqual(K_us[n2r_us["D"], n2r_us["E"]], 0.0)

    def test_lower_overlap(self):
        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 3,
                    ("E", "2"): 1, ("B", "2"): 2, ("C", "2"): 3, ("D", "2"): 4}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_molecule(cretention.dG, str_as_idf=True)

        dm, dc = cretention.dmolecules, cretention.dcollections
        self.assertIsInstance(dm, OrderedDict)
        self.assertIsInstance(dc, OrderedDict)

        K_ls, n2r_ls = RetentionGraph.order_tanimoto_kernel(d_moleculecut, type="lower", overlap=True)
        self.assertEqual(K_ls.shape, (5, 5))
        np.testing.assert_equal(np.diag(K_ls), np.ones(5))
        np.testing.assert_equal(K_ls, K_ls.T)
        self.assertTrue(np.all(K_ls <= 1.0))
        self.assertTrue(np.all(K_ls >= 0.0))

        # self.assertEqual(d_moleculecut["A"][1], dict([("B", 1), ("C", 2), ("D", 3)]))
        # self.assertEqual(d_moleculecut["B"][1], dict([("C", 1), ("D", 2)]))
        # self.assertEqual(d_moleculecut["C"][1], dict([("D", 1)]))
        # self.assertEqual(d_moleculecut["D"][1], dict())
        # self.assertEqual(d_moleculecut["E"][1], dict([("B", 1), ("C", 2), ("D", 3)]))
        self.assertEqual(K_ls[n2r_ls["A"], n2r_ls["B"]], 2. / 3.)
        self.assertEqual(K_ls[n2r_ls["A"], n2r_ls["C"]], 1. / 3.)
        self.assertEqual(K_ls[n2r_ls["B"], n2r_ls["C"]], 1. / 2.)
        self.assertEqual(K_ls[n2r_ls["B"], n2r_ls["D"]], 0.0)
        self.assertEqual(K_ls[n2r_ls["B"], n2r_ls["E"]], 2. / 3.)
        self.assertEqual(K_ls[n2r_ls["C"], n2r_ls["D"]], 0.0)
        self.assertEqual(K_ls[n2r_ls["C"], n2r_ls["E"]], 1. / 3.)

        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = {("A", "1"): 1, ("B", "1"): 2, ("C", "1"): 3, ("D", "1"): 9, ("E", "1"): 11,
                    ("F", "2"): 4, ("D", "2"): 6, ("C", "2"): 7,
                    ("S", "3"): 4, ("T", "3"): 5, ("U", "3"): 9}

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        d_moleculecut = cretention.upper_lower_set_molecule(cretention.dG, str_as_idf=True)

        dm, dc = cretention.dmolecules, cretention.dcollections
        self.assertIsInstance(dm, OrderedDict)
        self.assertIsInstance(dc, OrderedDict)

        K_ls, n2r_ls = RetentionGraph.order_tanimoto_kernel(d_moleculecut, type="lower", overlap=True)
        self.assertEqual(K_ls.shape, (9, 9))
        np.testing.assert_equal(np.diag(K_ls), np.ones(9))
        np.testing.assert_equal(K_ls, K_ls.T)
        self.assertTrue(np.all(K_ls <= 1.0))
        self.assertTrue(np.all(K_ls >= 0.0))

        # self.assertEqual(d_moleculecut["A"][1], dict([("B", 1), ("C", 2), ("D", 3), ("E", 4)]))
        # self.assertEqual(d_moleculecut["B"][1], dict([("C", 1), ("D", 2), ("E", 3)]))
        # self.assertEqual(d_moleculecut["C"][1], dict([("D", 1), ("E", 2)]))
        # self.assertEqual(d_moleculecut["D"][1], dict([("C", 1), ("E", 1)]))
        # self.assertEqual(d_moleculecut["E"][1], dict())
        # self.assertEqual(d_moleculecut["F"][1], dict([("D", 1), ("C", 2), ("E", 2)]))
        # self.assertEqual(d_moleculecut["S"][1], dict([("T", 1), ("U", 2)]))
        # self.assertEqual(d_moleculecut["T"][1], dict([("U", 1)]))
        # self.assertEqual(d_moleculecut["U"][1], dict())
        self.assertEqual(K_ls[n2r_ls["A"], n2r_ls["B"]], 3. / 4.)
        self.assertEqual(K_ls[n2r_ls["A"], n2r_ls["C"]], 2. / 4.)
        self.assertEqual(K_ls[n2r_ls["B"], n2r_ls["C"]], 2. / 3.)
        self.assertEqual(K_ls[n2r_ls["B"], n2r_ls["D"]], 2. / 3.)
        self.assertEqual(K_ls[n2r_ls["B"], n2r_ls["E"]], 0.0)
        self.assertEqual(K_ls[n2r_ls["B"], n2r_ls["T"]], 0.0)
        self.assertEqual(K_ls[n2r_ls["C"], n2r_ls["D"]], 1. / 3.)
        self.assertEqual(K_ls[n2r_ls["C"], n2r_ls["E"]], 0.0)
        self.assertEqual(K_ls[n2r_ls["C"], n2r_ls["U"]], 0.0)
        self.assertEqual(K_ls[n2r_ls["D"], n2r_ls["E"]], 0.0)
        self.assertEqual(K_ls[n2r_ls["T"], n2r_ls["U"]], 0.0)
        self.assertEqual(K_ls[n2r_ls["S"], n2r_ls["T"]], 1. / 2.)


if __name__ == "__main__":
    unittest.main()
