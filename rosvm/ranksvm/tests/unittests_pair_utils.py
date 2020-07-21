####
#
# The MIT License (MIT)
#
# Copyright 2017-2020 Eric Bach <eric.bach@aalto.fi>
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

# Import function to test
from ranksvm.pair_utils import get_pairs_multiple_datasets_SLOW, get_pairs_multiple_datasets
from ranksvm.rank_svm_cls import Labels


class TestPairGenerationSLOW(unittest.TestCase):
    def test_bordercases(self):
        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets_SLOW(Labels([1, 2, 3], [1, 2, 3]))

        self.assertEqual(0, len(pairs))
        self.assertEqual(0, len(signs))
        self.assertEqual(0, len(pdss))

        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets_SLOW(Labels([1, 2, 3], ["A", "B", "C"]))

        self.assertEqual(0, len(pairs))
        self.assertEqual(0, len(signs))
        self.assertEqual(0, len(pdss))

        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets_SLOW(Labels([2, 3, 2, 3], [1, 1, 2, 2]))

        self.assertEqual(2, len(pairs))
        self.assertEqual(2, len(signs))
        self.assertEqual(2, len(pdss))
        self.assertEqual([(0, 1), (2, 3)], pairs)
        self.assertEqual([-1, -1], signs)
        self.assertEqual([1, 2], pdss)

        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets_SLOW(Labels([2, 3, 2, 3], ["B", "B", "A", "A"]))

        self.assertEqual(2, len(pairs))
        self.assertEqual(2, len(signs))
        self.assertEqual(2, len(pdss))
        self.assertEqual([(0, 1), (2, 3)], pairs)
        self.assertEqual([-1, -1], signs)
        self.assertEqual(["B", "A"], pdss)

        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets_SLOW(Labels([2, 3, 1, 3, 3], [1, 1, 1, 2, 2]))

        self.assertEqual(3, len(pairs))
        self.assertEqual(3, len(signs))
        self.assertEqual(3, len(pdss))
        self.assertEqual([(0, 1), (0, 2), (1, 2)], pairs)
        self.assertEqual([-1, 1, 1], signs)
        self.assertEqual([1, 1, 1], pdss)

        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets_SLOW(Labels([2, 3, 1, 3, 3], ["A", "A", "A", "B", "B"]))

        self.assertEqual(3, len(pairs))
        self.assertEqual(3, len(signs))
        self.assertEqual(3, len(pdss))
        self.assertEqual([(0, 1), (0, 2), (1, 2)], pairs)
        self.assertEqual([-1, 1, 1], signs)
        self.assertEqual(["A", "A", "A"], pdss)

    def test_single_system_dataset(self):
        # ----------------------------------------------
        targets = Labels([10, 4, 6, 8, 2], ["A", "A", "A", "A", "A"])
        n_samples = len(targets)
        d_pairs_ref = {0: [],
                       1: [(1, 4), (1, 2), (2, 3), (0, 3)],
                       2: [(1, 4), (1, 2), (2, 3), (0, 3),
                           (2, 4), (1, 3), (0, 2)],
                       3: [(1, 4), (1, 2), (2, 3), (0, 3),
                           (2, 4), (1, 3), (0, 2),
                           (3, 4), (0, 1)],
                       4: [(1, 4), (1, 2), (2, 3), (0, 3),
                           (2, 4), (1, 3), (0, 2),
                           (3, 4), (0, 1),
                           (0, 4)]}
        d_pairs_ref[np.inf] = d_pairs_ref[4]
        d_signs_ref = {0: [],
                       1: [1, -1, -1, 1],
                       2: [1, -1, -1, 1,
                           1, -1, 1],
                       3: [1, -1, -1, 1,
                           1, -1, 1,
                           1, 1],
                       4: [1, -1, -1, 1,
                           1, -1, 1,
                           1, 1,
                           1]}
        d_signs_ref[np.inf] = d_signs_ref[4]

        for d in d_pairs_ref.keys():
            pairs, signs, pdss = get_pairs_multiple_datasets_SLOW(targets, d_upper=d)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            self.assertTrue(all([pds == "A" for pds in pdss]))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, _ = get_pairs_multiple_datasets_SLOW(targets, d_upper=np.inf)
        self.assertEqual((n_samples ** 2 - n_samples) / 2, len(pairs))

        # ----------------------------------------------
        d_pairs_ref = {5: [],
                       4: [(0, 4)],
                       3: [(0, 4), (3, 4), (0, 1)],
                       2: [(0, 4), (3, 4), (0, 1), (2, 4), (1, 3), (0, 2)],
                       1: [(0, 4), (3, 4), (0, 1), (2, 4), (1, 3), (0, 2), (1, 4), (1, 2), (2, 3), (0, 3)],
                       0: [(0, 4), (3, 4), (0, 1), (2, 4), (1, 3), (0, 2), (1, 4), (1, 2), (2, 3), (0, 3)]}
        d_signs_ref = {5: [],
                       4: [1],
                       3: [1, 1, 1],
                       2: [1, 1, 1, 1, -1, 1],
                       1: [1, 1, 1, 1, -1, 1, 1, -1, -1, 1],
                       0: [1, 1, 1, 1, -1, 1, 1, -1, -1, 1]}

        for d in d_pairs_ref.keys():
            pairs, signs, pdss = get_pairs_multiple_datasets_SLOW(targets, d_lower=d)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            self.assertTrue(all([pds == "A" for pds in pdss]))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, _ = get_pairs_multiple_datasets_SLOW(targets, d_lower=1)
        self.assertEqual((n_samples ** 2 - n_samples) / 2, len(pairs))

    def test_multiple_system_dataset(self):
        # ----------------------------------------------
        targets = Labels([6, 2, 3, 4, 1, 1, 10, 5, 12], ["A", "A", "A", "A", "A", "B", "B", "B", "B"])
        n_samples_A = 5
        n_samples_B = 4
        d_pairs_ref = {0: [],
                       1: [(1, 4), (1, 2), (2, 3), (0, 3), (5, 7), (6, 7), (6, 8)],
                       2: [(1, 4), (1, 2), (2, 3), (0, 3), (5, 7), (6, 7), (6, 8),
                           (2, 4), (1, 3), (0, 2), (5, 6), (7, 8)],
                       3: [(1, 4), (1, 2), (2, 3), (0, 3), (5, 7), (6, 7), (6, 8),
                           (2, 4), (1, 3), (0, 2), (5, 6), (7, 8),
                           (3, 4), (0, 1), (5, 8)],
                       4: [(1, 4), (1, 2), (2, 3), (0, 3), (5, 7), (6, 7), (6, 8),
                           (2, 4), (1, 3), (0, 2), (5, 6), (7, 8),
                           (3, 4), (0, 1), (5, 8),
                           (0, 4)]}
        d_pairs_ref[np.inf] = d_pairs_ref[4]
        d_signs_ref = {0: [],
                       1: [1, -1, -1, 1, -1, 1, -1],
                       2: [1, -1, -1, 1, -1, 1, -1,
                           1, -1, 1, -1, -1],
                       3: [1, -1, -1, 1, -1, 1, -1,
                           1, -1, 1, -1, -1,
                           1, 1, -1],
                       4: [1, -1, -1, 1, -1, 1, -1,
                           1, -1, 1, -1, -1,
                           1, 1, -1,
                           1]}
        d_signs_ref[np.inf] = d_signs_ref[4]

        for d in d_pairs_ref.keys():
            pairs, signs, _ = get_pairs_multiple_datasets_SLOW(targets, d_upper=d)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, pdss = get_pairs_multiple_datasets_SLOW(targets)
        max_n_pairs_A_ref = int((n_samples_A ** 2 - n_samples_A) / 2)
        max_n_pairs_B_ref = int((n_samples_B ** 2 - n_samples_B) / 2)
        self.assertEqual(max_n_pairs_A_ref + max_n_pairs_B_ref, len(pairs))
        self.assertEqual(["A"] * max_n_pairs_A_ref + ["B"] * max_n_pairs_B_ref, pdss)

        # ----------------------------------------------
        targets = Labels([1, 2, 3, 4, 1, 2, 1.5], ["A", "A", "A", "A", "B", "B", "B"])
        n_samples_A = 4
        n_samples_B = 3
        d_pairs_ref = {4: [],
                       3: [(0, 3)],
                       2: [(0, 3), (0, 2), (1, 3), (4, 5)],
                       1: [(0, 3), (0, 2), (1, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (5, 6)],
                       0: [(0, 3), (0, 2), (1, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (5, 6)]}
        d_signs_ref = {4: [],
                       3: [-1],
                       2: [-1, -1, -1, -1],
                       1: [-1, -1, -1,
                           -1, -1,
                           -1,
                           -1, -1,
                           1],
                       0: [-1, -1, -1,
                           -1, -1,
                           -1,
                           -1, -1,
                           1]}

        for d in d_pairs_ref.keys():
            pairs, signs, _ = get_pairs_multiple_datasets_SLOW(targets, d_lower=d)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, pdss = get_pairs_multiple_datasets_SLOW(targets)
        max_n_pairs_A_ref = int((n_samples_A ** 2 - n_samples_A) / 2)
        max_n_pairs_B_ref = int((n_samples_B ** 2 - n_samples_B) / 2)
        self.assertEqual(max_n_pairs_A_ref + max_n_pairs_B_ref, len(pairs))
        self.assertEqual(["A"] * max_n_pairs_A_ref + ["B"] * max_n_pairs_B_ref, pdss)


class TestPairGeneration(unittest.TestCase):
    def test_bordercases(self):
        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets(Labels([1, 2, 3], [1, 2, 3]))

        self.assertEqual(0, len(pairs))
        self.assertEqual(0, len(signs))
        self.assertEqual(0, len(pdss))

        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets(Labels([1, 2, 3], ["A", "B", "C"]))

        self.assertEqual(0, len(pairs))
        self.assertEqual(0, len(signs))
        self.assertEqual(0, len(pdss))

        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets(Labels([2, 3, 2, 3], [1, 1, 2, 2]))

        self.assertEqual(2, len(pairs))
        self.assertEqual(2, len(signs))
        self.assertEqual(2, len(pdss))
        self.assertEqual([(0, 1), (2, 3)], pairs)
        self.assertEqual([-1, -1], signs)
        self.assertEqual([1, 2], pdss)

        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets(Labels([2, 3, 2, 3], ["B", "B", "A", "A"]))

        self.assertEqual(2, len(pairs))
        self.assertEqual(2, len(signs))
        self.assertEqual(2, len(pdss))
        self.assertEqual([(0, 1), (2, 3)], pairs)
        self.assertEqual([-1, -1], signs)
        self.assertEqual(["B", "A"], pdss)

        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets(Labels([2, 3, 1, 3, 3], [1, 1, 1, 2, 2]))

        self.assertEqual(3, len(pairs))
        self.assertEqual(3, len(signs))
        self.assertEqual(3, len(pdss))
        self.assertEqual([(0, 1), (0, 2), (1, 2)], pairs)
        self.assertEqual([-1, 1, 1], signs)
        self.assertEqual([1, 1, 1], pdss)

        # ----------------------------------------------
        pairs, signs, pdss = get_pairs_multiple_datasets(Labels([2, 3, 1, 3, 3], ["A", "A", "A", "B", "B"]))

        self.assertEqual(3, len(pairs))
        self.assertEqual(3, len(signs))
        self.assertEqual(3, len(pdss))
        self.assertEqual([(0, 1), (0, 2), (1, 2)], pairs)
        self.assertEqual([-1, 1, 1], signs)
        self.assertEqual(["A", "A", "A"], pdss)

    def test_single_system_dataset_default_d_range(self):
        # ----------------------------------------------
        targets = Labels([10, 4, 6, 8, 2], ["A", "A", "A", "A", "A"])
        n_samples = len(targets)
        d_pairs_ref = {np.inf: [(1, 4), (1, 2), (2, 3), (0, 3),
                                (2, 4), (1, 3), (0, 2),
                                (3, 4), (0, 1),
                                (0, 4)]}
        d_signs_ref = {np.inf: [1, -1, -1, 1,
                                1, -1, 1,
                                1, 1,
                                1]}

        for d in d_pairs_ref.keys():
            pairs, signs, pdss = get_pairs_multiple_datasets(targets)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            self.assertTrue(all([pds == "A" for pds in pdss]))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, _ = get_pairs_multiple_datasets(targets)
        self.assertEqual((n_samples ** 2 - n_samples) / 2, len(pairs))

        # ----------------------------------------------
        d_pairs_ref = {1: [(0, 4), (3, 4), (0, 1), (2, 4), (1, 3), (0, 2), (1, 4), (1, 2), (2, 3), (0, 3)]}
        d_signs_ref = {1: [1, 1, 1, 1, -1, 1, 1, -1, -1, 1]}

        for d in d_pairs_ref.keys():
            pairs, signs, pdss = get_pairs_multiple_datasets(targets)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            self.assertTrue(all([pds == "A" for pds in pdss]))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, _ = get_pairs_multiple_datasets(targets)
        self.assertEqual((n_samples ** 2 - n_samples) / 2, len(pairs))

    def test_multiple_system_dataset_default_d_range(self):
        # ----------------------------------------------
        targets = Labels([6, 2, 3, 4, 1, 1, 10, 5, 12], ["A", "A", "A", "A", "A", "B", "B", "B", "B"])
        n_samples_A = 5
        n_samples_B = 4
        d_pairs_ref = {np.inf: [(1, 4), (1, 2), (2, 3), (0, 3), (5, 7), (6, 7), (6, 8),
                                (2, 4), (1, 3), (0, 2), (5, 6), (7, 8),
                                (3, 4), (0, 1), (5, 8),
                                (0, 4)]}
        d_signs_ref = {np.inf: [1, -1, -1, 1, -1, 1, -1,
                                1, -1, 1, -1, -1,
                                1, 1, -1,
                                1]}

        for d in d_pairs_ref.keys():
            pairs, signs, _ = get_pairs_multiple_datasets(targets)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, pdss = get_pairs_multiple_datasets(targets)
        max_n_pairs_A_ref = int((n_samples_A ** 2 - n_samples_A) / 2)
        max_n_pairs_B_ref = int((n_samples_B ** 2 - n_samples_B) / 2)
        self.assertEqual(max_n_pairs_A_ref + max_n_pairs_B_ref, len(pairs))
        self.assertEqual(["A"] * max_n_pairs_A_ref + ["B"] * max_n_pairs_B_ref, pdss)

        # ----------------------------------------------
        targets = Labels([1, 2, 3, 4, 1, 2, 1.5], ["A", "A", "A", "A", "B", "B", "B"])
        n_samples_A = 4
        n_samples_B = 3
        d_pairs_ref = {1: [(0, 3), (0, 2), (1, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (5, 6)]}
        d_signs_ref = {1: [-1, -1, -1,
                           -1, -1,
                           -1,
                           -1, -1,
                           1]}

        for d in d_pairs_ref.keys():
            pairs, signs, _ = get_pairs_multiple_datasets(targets)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, pdss = get_pairs_multiple_datasets(targets)
        max_n_pairs_A_ref = int((n_samples_A ** 2 - n_samples_A) / 2)
        max_n_pairs_B_ref = int((n_samples_B ** 2 - n_samples_B) / 2)
        self.assertEqual(max_n_pairs_A_ref + max_n_pairs_B_ref, len(pairs))
        self.assertEqual(["A"] * max_n_pairs_A_ref + ["B"] * max_n_pairs_B_ref, pdss)

    def test_single_system_dataset(self):
        # ----------------------------------------------
        targets = Labels([10, 4, 6, 8, 2], ["A", "A", "A", "A", "A"])
        n_samples = len(targets)
        d_pairs_ref = {0: [],
                       1: [(1, 4), (1, 2), (2, 3), (0, 3)],
                       2: [(1, 4), (1, 2), (2, 3), (0, 3),
                           (2, 4), (1, 3), (0, 2)],
                       3: [(1, 4), (1, 2), (2, 3), (0, 3),
                           (2, 4), (1, 3), (0, 2),
                           (3, 4), (0, 1)],
                       4: [(1, 4), (1, 2), (2, 3), (0, 3),
                           (2, 4), (1, 3), (0, 2),
                           (3, 4), (0, 1),
                           (0, 4)]}
        d_signs_ref = {0: [],
                       1: [1, -1, -1, 1],
                       2: [1, -1, -1, 1,
                           1, -1, 1],
                       3: [1, -1, -1, 1,
                           1, -1, 1,
                           1, 1],
                       4: [1, -1, -1, 1,
                           1, -1, 1,
                           1, 1,
                           1]}

        for d in d_pairs_ref.keys():
            pairs, signs, pdss = get_pairs_multiple_datasets(targets, d_upper=d)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            self.assertTrue(all([pds == "A" for pds in pdss]))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, _ = get_pairs_multiple_datasets(targets, d_upper=np.inf)
        self.assertEqual((n_samples ** 2 - n_samples) / 2, len(pairs))

        # ----------------------------------------------
        d_pairs_ref = {5: [],
                       4: [(0, 4)],
                       3: [(0, 4), (3, 4), (0, 1)],
                       2: [(0, 4), (3, 4), (0, 1), (2, 4), (1, 3), (0, 2)],
                       1: [(0, 4), (3, 4), (0, 1), (2, 4), (1, 3), (0, 2), (1, 4), (1, 2), (2, 3), (0, 3)],
                       0: [(0, 4), (3, 4), (0, 1), (2, 4), (1, 3), (0, 2), (1, 4), (1, 2), (2, 3), (0, 3)]}
        d_signs_ref = {5: [],
                       4: [1],
                       3: [1, 1, 1],
                       2: [1, 1, 1, 1, -1, 1],
                       1: [1, 1, 1, 1, -1, 1, 1, -1, -1, 1],
                       0: [1, 1, 1, 1, -1, 1, 1, -1, -1, 1]}

        for d in d_pairs_ref.keys():
            pairs, signs, pdss = get_pairs_multiple_datasets(targets, d_lower=d)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            self.assertTrue(all([pds == "A" for pds in pdss]))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, _ = get_pairs_multiple_datasets(targets, d_lower=1)
        self.assertEqual((n_samples ** 2 - n_samples) / 2, len(pairs))

    def test_multiple_system_dataset(self):
        # ----------------------------------------------
        targets = Labels([6, 2, 3, 4, 1, 1, 10, 5, 12], ["A", "A", "A", "A", "A", "B", "B", "B", "B"])
        n_samples_A = 5
        n_samples_B = 4
        d_pairs_ref = {0: [],
                       1: [(1, 4), (1, 2), (2, 3), (0, 3), (5, 7), (6, 7), (6, 8)],
                       2: [(1, 4), (1, 2), (2, 3), (0, 3), (5, 7), (6, 7), (6, 8),
                           (2, 4), (1, 3), (0, 2), (5, 6), (7, 8)],
                       3: [(1, 4), (1, 2), (2, 3), (0, 3), (5, 7), (6, 7), (6, 8),
                           (2, 4), (1, 3), (0, 2), (5, 6), (7, 8),
                           (3, 4), (0, 1), (5, 8)],
                       4: [(1, 4), (1, 2), (2, 3), (0, 3), (5, 7), (6, 7), (6, 8),
                           (2, 4), (1, 3), (0, 2), (5, 6), (7, 8),
                           (3, 4), (0, 1), (5, 8),
                           (0, 4)]}
        d_signs_ref = {0: [],
                       1: [1, -1, -1, 1, -1, 1, -1],
                       2: [1, -1, -1, 1, -1, 1, -1,
                           1, -1, 1, -1, -1],
                       3: [1, -1, -1, 1, -1, 1, -1,
                           1, -1, 1, -1, -1,
                           1, 1, -1],
                       4: [1, -1, -1, 1, -1, 1, -1,
                           1, -1, 1, -1, -1,
                           1, 1, -1,
                           1]}

        for d in d_pairs_ref.keys():
            pairs, signs, _ = get_pairs_multiple_datasets(targets, d_upper=d)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, pdss = get_pairs_multiple_datasets(targets)
        max_n_pairs_A_ref = int((n_samples_A ** 2 - n_samples_A) / 2)
        max_n_pairs_B_ref = int((n_samples_B ** 2 - n_samples_B) / 2)
        self.assertEqual(max_n_pairs_A_ref + max_n_pairs_B_ref, len(pairs))
        self.assertEqual(["A"] * max_n_pairs_A_ref + ["B"] * max_n_pairs_B_ref, pdss)

        # ----------------------------------------------
        targets = Labels([1, 2, 3, 4, 1, 2, 1.5], ["A", "A", "A", "A", "B", "B", "B"])
        n_samples_A = 4
        n_samples_B = 3
        d_pairs_ref = {4: [],
                       3: [(0, 3)],
                       2: [(0, 3), (0, 2), (1, 3), (4, 5)],
                       1: [(0, 3), (0, 2), (1, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (5, 6)],
                       0: [(0, 3), (0, 2), (1, 3),
                           (1, 2), (1, 3),
                           (2, 3),
                           (4, 5), (4, 6),
                           (5, 6)]}
        d_signs_ref = {4: [],
                       3: [-1],
                       2: [-1, -1, -1, -1],
                       1: [-1, -1, -1,
                           -1, -1,
                           -1,
                           -1, -1,
                           1],
                       0: [-1, -1, -1,
                           -1, -1,
                           -1,
                           -1, -1,
                           1]}

        for d in d_pairs_ref.keys():
            pairs, signs, _ = get_pairs_multiple_datasets(targets, d_lower=d)

            self.assertEqual(len(d_pairs_ref[d]), len(pairs))
            self.assertEqual(len(d_signs_ref[d]), len(signs))
            for pair, sign in zip(d_pairs_ref[d], d_signs_ref[d]):
                self.assertIn(pair, pairs)
                self.assertEqual(sign, signs[pairs.index(pair)])

        pairs, _, pdss = get_pairs_multiple_datasets(targets)
        max_n_pairs_A_ref = int((n_samples_A ** 2 - n_samples_A) / 2)
        max_n_pairs_B_ref = int((n_samples_B ** 2 - n_samples_B) / 2)
        self.assertEqual(max_n_pairs_A_ref + max_n_pairs_B_ref, len(pairs))
        self.assertEqual(["A"] * max_n_pairs_A_ref + ["B"] * max_n_pairs_B_ref, pdss)


if __name__ == '__main__':
    unittest.main()
