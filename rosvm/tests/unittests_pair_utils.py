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

from scipy.stats import rankdata

from collections import OrderedDict
from ranksvm.retentiongraph_cls import RetentionGraph

# Import function to test
from ranksvm.pair_utils import get_pairs_from_order_graph, get_pairs_multiple_datasets, get_pairs_single_dataset


class TestSingleSystem(unittest.TestCase):
    def test_bordercases(self):
        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 2), (("M2", "A"), 2), (("M3", "A"), 2)])
        pairs, _ = get_pairs_single_dataset(list(d_target.values()), d_lower=0, d_upper=np.inf)

        self.assertEqual(len(pairs), 0)

    def test_get_pairs(self):
        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 10), (("M2", "A"), 4), (("M3", "A"), 6), (("M4", "A"), 8),
                                (("M5", "A"), 2)])

        d_pairs_ref = {0: [],
                       1: [(4, 1), (1, 2), (2, 3), (3, 0)],
                       2: [(4, 1), (1, 2), (2, 3), (3, 0),
                           (4, 2), (1, 3), (2, 0)],
                       3: [(4, 1), (1, 2), (2, 3), (3, 0),
                           (4, 2), (1, 3), (2, 0),
                           (4, 3), (1, 0)],
                       4: [(4, 1), (1, 2), (2, 3), (3, 0),
                           (4, 2), (1, 3), (2, 0),
                           (4, 3), (1, 0),
                           (4, 0)]}

        for d in d_pairs_ref.keys():
            pairs, _ = get_pairs_single_dataset(list(d_target.values()), d_lower=0, d_upper=d)

            self.assertEqual(len(pairs), len(d_pairs_ref[d]))
            for pair in d_pairs_ref[d]:
                self.assertIn(pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 10), (("M2", "A"), 4), (("M3", "A"), 6), (("M4", "A"), 8),
                                (("M5", "A"), 2)])

        d_pairs_ref = {5: [],
                       4: [(4, 0)],
                       3: [(4, 0), (4, 3), (1, 0)],
                       2: [(4, 0), (4, 3), (1, 0), (4, 2), (1, 3), (2, 0)],
                       1: [(4, 0), (4, 3), (1, 0), (4, 2), (1, 3), (2, 0), (4, 1), (1, 2), (2, 3), (3, 0)],
                       0: [(4, 0), (4, 3), (1, 0), (4, 2), (1, 3), (2, 0), (4, 1), (1, 2), (2, 3), (3, 0)]}

        for d in d_pairs_ref.keys():
            pairs, _ = get_pairs_single_dataset(list(d_target.values()), d_lower=d, d_upper=np.inf)

            self.assertEqual(len(pairs), len(d_pairs_ref[d]))
            for pair in d_pairs_ref[d]:
                self.assertIn(pair, pairs)

    def test_get_pairwise_confidences(self):
        # Mol:      A, B, C, D, E, F,  G
        targets = [10, 4, 6, 8, 2, 1, 11]
        ranks = rankdata(targets, method="dense")

        # ----------------------------------------------
        pairs, pair_confs = get_pairs_single_dataset(targets)
        self.assertEqual(len(pairs), np.arange(1, len(targets)).sum())
        np.testing.assert_equal(pair_confs, np.ones((len(pairs),)))

        # ----------------------------------------------
        pairs, pair_confs = get_pairs_single_dataset(targets, pw_conf_fun=lambda x, y, **kwargs: y - x)
        for (i, j), conf in zip(pairs, pair_confs):
            self.assertEqual(conf, ranks[j] - ranks[i])

        # ----------------------------------------------
        # Tanimoto kernel on the upper set
        pairs, pair_confs = get_pairs_single_dataset(
            targets, pw_conf_fun=lambda x, y, **kwargs: (x - 1) / (y - 1))
        self.assertIn((2, 0), pairs)
        self.assertEqual(pair_confs[pairs.index((2, 0))], 3. / 5.)
        self.assertIn((5, 4), pairs)
        self.assertEqual(pair_confs[pairs.index((5, 4))], 0)
        self.assertIn((0, 6), pairs)
        self.assertEqual(pair_confs[pairs.index((0, 6))], 5. / 6.)
        self.assertIn((4, 2), pairs)
        self.assertEqual(pair_confs[pairs.index((4, 2))], 1. / 3.)

        # ----------------------------------------------
        # Tanimoto kernel on the lower set
        def pw_conf_fun(x, y, **kwargs):
            max_rank = kwargs["max_rank"]
            return (max_rank - y) / (max_rank - x)
        pairs, pair_confs = get_pairs_single_dataset(targets, pw_conf_fun=pw_conf_fun)
        self.assertIn((2, 0), pairs)
        self.assertEqual(pair_confs[pairs.index((2, 0))], 1. / 3.)
        self.assertIn((5, 4), pairs)
        self.assertEqual(pair_confs[pairs.index((5, 4))], 5. / 6.)
        self.assertIn((0, 6), pairs)
        self.assertEqual(pair_confs[pairs.index((0, 6))], 0)
        self.assertIn((4, 2), pairs)
        self.assertEqual(pair_confs[pairs.index((4, 2))], 3. / 5.)


class TestMultipleSystems(unittest.TestCase):
    def test_bordercases(self):
        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M1", "B"), 2), (("M1", "C"), 3)])
        m_target = np.array([list(d_target.values()), [1, 2, 3]]).T
        pairs = get_pairs_multiple_datasets(m_target, d_lower=0, d_upper=np.inf)

        self.assertEqual(len(pairs), 0)

        # ----------------------------------------------
        d_target = OrderedDict([(("M2", "A"), 2), (("M3", "A"), 3), (("M3", "B"), 2), (("M2", "B"), 3)])
        m_target = np.array([list(d_target.values()), [1, 1, 2, 2]]).T
        pairs = get_pairs_multiple_datasets(m_target, d_lower=0, d_upper=np.inf)

        self.assertEqual(len(pairs), 2)

    def test_get_pairs(self):
        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 5), (("M2", "A"), 2), (("M3", "A"), 3), (("M4", "A"), 4),
                                (("M5", "A"), 1),
                                (("M5", "B"), 1), (("M9", "B"), 10), (("M7", "B"), 5), (("M1", "B"), 12)])
        m_target = np.array([list(d_target.values()), [1, 1, 1, 1, 1, 2, 2, 2, 2]]).T

        d_pairs_ref = {0: [],
                       1: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8)],
                       2: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8),
                           (4, 2), (1, 3), (2, 0), (5, 6), (7, 8)],
                       3: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8),
                           (4, 2), (1, 3), (2, 0), (5, 6), (7, 8),
                           (4, 3), (1, 0), (5, 8)],
                       4: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8),
                           (4, 2), (1, 3), (2, 0), (5, 6), (7, 8),
                           (4, 3), (1, 0), (5, 8),
                           (4, 0)]}

        for d in d_pairs_ref.keys():
            pairs = get_pairs_multiple_datasets(m_target, d_lower=0, d_upper=d)

            self.assertEqual(len(pairs), len(d_pairs_ref[d]))
            for pair in d_pairs_ref[d]:
                self.assertIn(pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2), (("M7", "B"), 1.5)])
        m_target = np.array([list(d_target.values()), [1, 1, 1, 1, 2, 2, 2]]).T

        d_pairs_ref = {4: [],
                       3: [(0, 3)],
                       2: [(0, 3), (0, 2), (1, 3), (4, 5)],
                       1: [(0, 1), (0, 2), (0, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (6, 5)],
                       0: [(0, 1), (0, 2), (0, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (6, 5)]}

        for d in d_pairs_ref.keys():
            pairs = get_pairs_multiple_datasets(m_target, d_lower=d, d_upper=np.inf)
            self.assertEqual(len(pairs), len(d_pairs_ref[d]))

            for pair in d_pairs_ref[d]:
                self.assertIn(pair, pairs)


class TestOrderGraph(unittest.TestCase):
    def test_simplecases(self):
        cretention = RetentionGraph()

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M1", "B"), 1), (("M5", "B"), 2)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        for allow_overlap in [True, False]:
            pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=allow_overlap,
                                               d_lower=0, d_upper=np.inf)
            self.assertEqual(len(pairs), 11)
            self.assertIn((0, 1), pairs)
            self.assertIn((0, 2), pairs)
            self.assertIn((0, 3), pairs)
            self.assertIn((0, 5), pairs)
            self.assertIn((1, 2), pairs)
            self.assertIn((1, 3), pairs)
            self.assertIn((2, 3), pairs)
            self.assertIn((4, 1), pairs)
            self.assertIn((4, 2), pairs)
            self.assertIn((4, 3), pairs)
            self.assertIn((4, 5), pairs)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2),
                                (("M7", "B"), 3)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        for allow_overlap in [True, False]:
            pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=allow_overlap,
                                               d_lower=0, d_upper=np.inf)
            self.assertEqual(len(pairs), 18)
            self.assertIn((0, 1), pairs)
            self.assertIn((0, 2), pairs)
            self.assertIn((0, 3), pairs)
            self.assertIn((0, 4), pairs)
            self.assertIn((0, 5), pairs)
            self.assertIn((0, 6), pairs)
            self.assertIn((1, 2), pairs)
            self.assertIn((1, 3), pairs)
            self.assertIn((1, 5), pairs)
            self.assertIn((1, 6), pairs)
            self.assertIn((2, 3), pairs)
            self.assertIn((2, 6), pairs)
            self.assertIn((4, 2), pairs)
            self.assertIn((4, 3), pairs)
            self.assertIn((4, 5), pairs)
            self.assertIn((4, 6), pairs)
            self.assertIn((5, 3), pairs)
            self.assertIn((5, 6), pairs)

    def test_bordercases(self):
        cretention = RetentionGraph()

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M1", "B"), 2), (("M1", "C"), 3)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        for allow_overlap in [True, False]:
            pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=allow_overlap,
                                               d_lower=0, d_upper=np.inf)
            self.assertEqual(len(pairs), 0)

        # ----------------------------------------------
        d_target = OrderedDict([(("M2", "A"), 2), (("M3", "A"), 3),
                                (("M3", "B"), 2), (("M2", "B"), 3)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=True,
                                           d_lower=0, d_upper=np.inf)
        self.assertEqual(len(pairs), 8)

        pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=False,
                                           d_lower=0, d_upper=np.inf)
        self.assertEqual(len(pairs), 0)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=True, d_lower=0, d_upper=0)
        self.assertEqual(len(pairs), 0)

        pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=False, d_lower=0, d_upper=0)
        self.assertEqual(len(pairs), 0)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=True, d_lower=np.inf, d_upper=np.inf)
        self.assertEqual(len(pairs), 0)

        pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=False, d_lower=np.inf, d_upper=np.inf)
        self.assertEqual(len(pairs), 0)

    def test_overlap(self):
        cretention = RetentionGraph()

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=True, d_lower=0, d_upper=np.inf)

        self.assertEqual(len(pairs), 17)
        self.assertIn((0, 1), pairs)
        self.assertIn((0, 2), pairs)
        self.assertIn((0, 3), pairs)
        self.assertIn((0, 5), pairs)
        self.assertIn((0, 4), pairs)
        self.assertIn((1, 2), pairs)
        self.assertIn((1, 3), pairs)
        self.assertIn((1, 5), pairs)
        self.assertIn((2, 3), pairs)
        self.assertIn((2, 4), pairs)
        self.assertIn((2, 1), pairs)
        self.assertIn((5, 4), pairs)
        self.assertIn((5, 1), pairs)
        self.assertIn((5, 3), pairs)
        self.assertIn((4, 2), pairs)
        self.assertIn((4, 5), pairs)
        self.assertIn((4, 3), pairs)

        pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=False, d_lower=0, d_upper=np.inf)

        self.assertEqual(len(pairs), 9)
        self.assertIn((0, 1), pairs)
        self.assertIn((0, 2), pairs)
        self.assertIn((0, 3), pairs)
        self.assertIn((0, 5), pairs)
        self.assertIn((0, 4), pairs)
        self.assertIn((1, 3), pairs)
        self.assertIn((2, 3), pairs)
        self.assertIn((5, 3), pairs)
        self.assertIn((4, 3), pairs)

    def test_d(self):
        cretention = RetentionGraph()

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2),
                                (("M7", "B"), 3)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {0: [],
                       1: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (4, 5), (4, 2), (5, 3), (5, 6)],
                       2: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (4, 5), (4, 2), (5, 3), (5, 6),
                           (0, 2), (0, 5), (1, 6), (1, 3), (4, 6), (4, 3)],
                       3: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (4, 5), (4, 2), (5, 3), (5, 6),
                           (0, 2), (0, 5), (1, 6), (1, 3), (4, 6), (4, 3),
                           (0, 3), (0, 6)]}

        for allow_overlap in [True, False]:
            for d in d_pairs_ref.keys():
                pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=allow_overlap,
                                                   d_lower=0, d_upper=d)

                self.assertEqual(len(pairs), len(d_pairs_ref[d]))

                for pair in d_pairs_ref[d]:
                    self.assertIn(pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {0: [],
                       1: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 4), (2, 1), (5, 4), (5, 3), (5, 1), (4, 2),
                           (4, 5)],
                       2: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 4), (2, 1), (5, 4), (5, 3), (5, 1), (4, 2),
                           (4, 5), (0, 2), (0, 5), (1, 3), (4, 3)],
                       3: [(0, 1), (0, 4), (1, 2), (1, 5), (2, 3), (2, 4), (2, 1), (5, 4), (5, 3), (5, 1), (4, 2),
                           (4, 5), (0, 2), (0, 5), (1, 3), (4, 3), (0, 3)]}

        for d in d_pairs_ref.keys():
            pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=True,
                                               d_lower=0, d_upper=d)

            self.assertEqual(len(pairs), len(d_pairs_ref[d]))

            for pair in d_pairs_ref[d]:
                self.assertIn(pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {0: [],
                       1: [(0, 1), (0, 4), (2, 3), (5, 3)],
                       2: [(0, 1), (0, 4), (2, 3), (5, 3),
                           (0, 2), (0, 5), (1, 3), (4, 3)],
                       3: [(0, 1), (0, 4), (2, 3), (5, 3),
                           (0, 2), (0, 5), (1, 3), (4, 3),
                           (0, 3)]}

        for d in d_pairs_ref.keys():
            pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=False, d_lower=0,
                                               d_upper=d)

            self.assertEqual(len(pairs), len(d_pairs_ref[d]))

            for pair in d_pairs_ref[d]:
                self.assertIn(pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 2), (("M3", "B"), 1)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {4: [],
                       3: [(0, 3)],
                       2: [(0, 2), (0, 3), (0, 5), (1, 3), (4, 3)],
                       1: [(0, 1), (0, 4), (2, 3), (5, 3),
                           (0, 2), (0, 5), (1, 3), (4, 3),
                           (0, 3)],
                       0: [(0, 1), (0, 4), (2, 3), (5, 3),
                           (0, 2), (0, 5), (1, 3), (4, 3),
                           (0, 3)]}

        for d in d_pairs_ref.keys():
            pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=False, d_lower=d,
                                               d_upper=np.inf)

            self.assertEqual(len(pairs), len(d_pairs_ref[d]))

            for pair in d_pairs_ref[d]:
                self.assertIn(pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2), (("M7", "B"), 1.5)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {4: [],
                       3: [(0, 3)],
                       2: [(0, 3), (0, 5), (0, 2), (0, 6),
                           (1, 3),
                           (4, 3),
                           (6, 3)],
                       1: [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                           (1, 2), (1, 3), (1, 5), (1, 6),
                           (2, 3),
                           (4, 2), (4, 3), (4, 5), (4, 6),
                           (5, 3),
                           (6, 2), (6, 3), (6, 5)],
                       0: [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                           (1, 2), (1, 3), (1, 5), (1, 6),
                           (2, 3),
                           (4, 2), (4, 3), (4, 5), (4, 6),
                           (5, 3),
                           (6, 2), (6, 3), (6, 5)]}

        for allow_overlap in [True, False]:
            for d in d_pairs_ref.keys():
                pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=allow_overlap, d_lower=d,
                                                   d_upper=np.inf)

                self.assertEqual(len(pairs), len(d_pairs_ref[d]))

                for pair in d_pairs_ref[d]:
                    self.assertIn(pair, pairs)

    def test_ireversed(self):
        cretention = RetentionGraph()

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2),
                                (("M7", "B"), 3)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph(ireverse=0)
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        pairs_ref = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (0, 2), (1, 3), (4, 6), (0, 3)]
        pairs_notin_ref = [(1, 6), (1, 5), (0, 5), (0, 4)]

        for allow_overlap in [True, False]:
            pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=allow_overlap,
                                               d_lower=0, d_upper=np.inf)

            self.assertEqual(len(pairs), len(pairs_ref))

            for pair in pairs_ref:
                self.assertIn(pair, pairs)

            for pair in pairs_notin_ref:
                self.assertNotIn(pair, pairs)

    def test_equal_to_simple_function_in_single_system_case(self):
        cretention = RetentionGraph()

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 10), (("M2", "A"), 4), (("M3", "A"), 6), (("M4", "A"), 8),
                                (("M5", "A"), 2)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {0: [],
                       1: [(4, 1), (1, 2), (2, 3), (3, 0)],
                       2: [(4, 1), (1, 2), (2, 3), (3, 0),
                           (4, 2), (1, 3), (2, 0)],
                       3: [(4, 1), (1, 2), (2, 3), (3, 0),
                           (4, 2), (1, 3), (2, 0),
                           (4, 3), (1, 0)],
                       4: [(4, 1), (1, 2), (2, 3), (3, 0),
                           (4, 2), (1, 3), (2, 0),
                           (4, 3), (1, 0),
                           (4, 0)]}

        for d in d_pairs_ref.keys():
            pairs_og = get_pairs_from_order_graph(cretention, keys, allow_overlap=True,
                                                  d_lower=0, d_upper=d)
            pairs, _ = get_pairs_single_dataset(list(d_target.values()), d_lower=0, d_upper=d)

            self.assertEqual(len(pairs_og), len(d_pairs_ref[d]))
            self.assertEqual(len(pairs), len(d_pairs_ref[d]))

            for pair in d_pairs_ref[d]:
                self.assertIn(pair, pairs_og)
                self.assertIn(pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 10), (("M2", "A"), 4), (("M3", "A"), 6), (("M4", "A"), 8),
                                (("M5", "A"), 2)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph()
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {5: [],
                       4: [(4, 0)],
                       3: [(4, 0), (4, 3), (1, 0)],
                       2: [(4, 0), (4, 3), (1, 0), (4, 2), (1, 3), (2, 0)],
                       1: [(4, 0), (4, 3), (1, 0), (4, 2), (1, 3), (2, 0), (4, 1), (1, 2), (2, 3), (3, 0)],
                       0: [(4, 0), (4, 3), (1, 0), (4, 2), (1, 3), (2, 0), (4, 1), (1, 2), (2, 3), (3, 0)]}

        for d in d_pairs_ref.keys():
            pairs_og = get_pairs_from_order_graph(cretention, keys, allow_overlap=True,
                                                  d_lower=d, d_upper=np.inf)
            pairs, _ = get_pairs_single_dataset(list(d_target.values()), d_lower=d, d_upper=np.inf)

            self.assertEqual(len(pairs_og), len(d_pairs_ref[d]))
            self.assertEqual(len(pairs), len(d_pairs_ref[d]))

            for pair in d_pairs_ref[d]:
                self.assertIn(pair, pairs_og)
                self.assertIn(pair, pairs)

    def test_equal_to_simple_function_in_multiple_system_case(self):
        # ----------------------------------------------
        cretention = RetentionGraph()
        d_target = OrderedDict([(("M1", "A"), 5), (("M2", "A"), 2), (("M3", "A"), 3), (("M4", "A"), 4),
                                (("M5", "A"), 1),
                                (("M5", "B"), 1), (("M9", "B"), 10), (("M7", "B"), 5), (("M1", "B"), 12)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph(ireverse=0)
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {0: [],
                       1: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8)],
                       2: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8),
                           (4, 2), (1, 3), (2, 0), (5, 6), (7, 8)],
                       3: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8),
                           (4, 2), (1, 3), (2, 0), (5, 6), (7, 8),
                           (4, 3), (1, 0), (5, 8)],
                       4: [(4, 1), (1, 2), (2, 3), (3, 0), (5, 7), (7, 6), (6, 8),
                           (4, 2), (1, 3), (2, 0), (5, 6), (7, 8),
                           (4, 3), (1, 0), (5, 8),
                           (4, 0)]}
        for d in d_pairs_ref.keys():
            pairs_og = get_pairs_from_order_graph(cretention, keys, allow_overlap=True,
                                                  d_lower=0, d_upper=d)
            m_target = np.array([list(d_target.values()), [1, 1, 1, 1, 1, 2, 2, 2, 2]]).T
            pairs = get_pairs_multiple_datasets(m_target, d_lower=0, d_upper=d)

            self.assertEqual(len(pairs_og), len(d_pairs_ref[d]))
            self.assertEqual(len(pairs), len(d_pairs_ref[d]))

            for pair in d_pairs_ref[d]:
                self.assertIn(pair, pairs_og)
                self.assertIn(pair, pairs)

        # ----------------------------------------------
        d_target = OrderedDict([(("M1", "A"), 1), (("M2", "A"), 2), (("M3", "A"), 3),
                                (("M4", "A"), 4), (("M2", "B"), 1), (("M3", "B"), 2), (("M7", "B"), 1.5)])
        keys = list(d_target.keys())

        cretention.load_data_from_target(d_target)
        cretention.make_digraph(ireverse=False)
        cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
        cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

        d_pairs_ref = {4: [],
                       3: [(0, 3)],
                       2: [(0, 3), (0, 2), (1, 3), (4, 5)],
                       1: [(0, 1), (0, 2), (0, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (6, 5)],
                       0: [(0, 1), (0, 2), (0, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (6, 5)]}

        for allow_overlap in [True, False]:
            for d in d_pairs_ref.keys():
                pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=allow_overlap, d_lower=d,
                                                   d_upper=np.inf)

                m_target = np.array([list(d_target.values()), [1, 1, 1, 1, 2, 2, 2]]).T
                pairs_sf = get_pairs_multiple_datasets(m_target, d_lower=d, d_upper=np.inf)

                self.assertEqual(len(pairs), len(d_pairs_ref[d]))
                self.assertEqual(len(pairs_sf), len(d_pairs_ref[d]))

                for pair in d_pairs_ref[d]:
                    self.assertIn(pair, pairs)
                    self.assertIn(pair, pairs_sf)


if __name__ == '__main__':
    unittest.main()
