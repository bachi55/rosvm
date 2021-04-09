####
#
# The MIT License (MIT)
#
# Copyright 2020 Eric Bach <eric.bach@aalto.fi>
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
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint, GetMorganFingerprintAsBitVect
from scipy.sparse import isspmatrix_csr
from sklearn.exceptions import NotFittedError
from sklearn.base import clone

from rosvm.feature_extraction.featurizer_cls import CircularFPFeaturizer, EStateIndFeaturizer


class TestEStateIndFeaturizer(unittest.TestCase):
    def setUp(self) -> None:
        self.smis = ["CC(=O)C1=CC2=C(OC(C)(C)[C@@H](O)[C@@H]2O)C=C1",
                     "C1COC2=CC=CC=C2C1",
                     "O=C(CCc1ccc(O)cc1)c1c(O)cc(O)c(C2OC(CO)C(O)C(O)C2O)c1O",
                     "O=c1c(OC2OC(CO)C(O)C(O)C2O)c(-c2ccc(OC3OC(CO)C(O)C(O)C3O)c(O)c2)oc2cc(O)cc(O)c12",
                     "O=C(O)C1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O",
                     "Oc1cc(O)c2c(c1)OC1(c3ccc(O)c(O)c3)Oc3cc(O)c4c(c3C2C1O)OC(c1ccc(O)c(O)c1)C(O)C4",
                     "COc1cc(O)c2c(=O)c(O)c(-c3ccc(O)c(O)c3)oc2c1",
                     "CC1OC(O)C(O)C(O)C1O",
                     "Cc1cc2nc3c(O)nc(=O)nc-3n(CC(O)C(O)C(O)CO)c2cc1C",
                     "O=C(C=Cc1ccc(O)c(O)c1)OC(Cc1ccc(O)c(O)c1)C(=O)O",
                     "COc1cc(O)c2c(c1)OC(c1ccc(O)cc1)CC2=O",
                     "C=CC(C)(O)CCC1C(C)(O)CCC2C(C)(C)CCCC21C",
                     "COc1cc2ccc(=O)oc2cc1O",
                     "NCCc1c[nH]c2ccc(O)cc12"]
        self.mols = [MolFromSmiles(smi) for smi in self.smis]
        self.n_mols = len(self.mols)

    def test__fingerprinter(self):
        fps_from_smis = EStateIndFeaturizer().fit_transform(self.smis)
        fps_from_mols = EStateIndFeaturizer().fit_transform(self.mols)

        self.assertEqual((self.n_mols, 79), fps_from_smis.shape)
        self.assertEqual((self.n_mols, 79), fps_from_mols.shape)
        np.testing.assert_equal(fps_from_smis, fps_from_mols)

    def test__parallel_fingerprinter(self):
        fps_from_sgl = EStateIndFeaturizer(n_jobs=1).fit_transform(self.smis)
        fps_from_par = EStateIndFeaturizer(n_jobs=4).fit_transform(self.smis)

        self.assertEqual((self.n_mols, 79), fps_from_par.shape)
        np.testing.assert_equal(fps_from_sgl, fps_from_par)


class TestCircularFPFeaturizer(unittest.TestCase):
    def setUp(self) -> None:
        self.smis = ["CC(=O)C1=CC2=C(OC(C)(C)[C@@H](O)[C@@H]2O)C=C1",
                     "C1COC2=CC=CC=C2C1",
                     "O=C(CCc1ccc(O)cc1)c1c(O)cc(O)c(C2OC(CO)C(O)C(O)C2O)c1O",
                     "O=c1c(OC2OC(CO)C(O)C(O)C2O)c(-c2ccc(OC3OC(CO)C(O)C(O)C3O)c(O)c2)oc2cc(O)cc(O)c12",
                     "O=C(O)C1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O",
                     "Oc1cc(O)c2c(c1)OC1(c3ccc(O)c(O)c3)Oc3cc(O)c4c(c3C2C1O)OC(c1ccc(O)c(O)c1)C(O)C4",
                     "COc1cc(O)c2c(=O)c(O)c(-c3ccc(O)c(O)c3)oc2c1",
                     "CC1OC(O)C(O)C(O)C1O",
                     "Cc1cc2nc3c(O)nc(=O)nc-3n(CC(O)C(O)C(O)CO)c2cc1C",
                     "O=C(C=Cc1ccc(O)c(O)c1)OC(Cc1ccc(O)c(O)c1)C(=O)O",
                     "COc1cc(O)c2c(c1)OC(c1ccc(O)cc1)CC2=O",
                     "C=CC(C)(O)CCC1C(C)(O)CCC2C(C)(C)CCCC21C",
                     "COc1cc2ccc(=O)oc2cc1O",
                     "NCCc1c[nH]c2ccc(O)cc12"]
        self.mols = [MolFromSmiles(smi) for smi in self.smis]
        self.n_mols = len(self.mols)

    def test__error_when_parsing_smiles(self) -> None:
        with self.assertRaises(RuntimeError):
            CircularFPFeaturizer().fit_transform([
                "O=C(O)C1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O",
                "Oc1cc(O)c2c(c1)OC1(c3ccc(O)c(O)c3)Oc3cc(O)c4c(c3C2C1O)OC(c1ccc(O)c(O)c1)C(O)C4",
                "COc1cc(O)c2c(=O)c(O)c(-c3ccc(O)c(O)c3)oc2c1",
                "CaC1asfOC(O)C(O)C(O)C1O"])

    def test__count_and_filter_hashes(self) -> None:
        d = [
            {"A": 0, "B": 0, "C": 0, "D": 0},
            {"A": 0 ,        "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {"A": 0, "B": 0, "C": 0, "D": 0},
            {                        "D": 0}
        ]  # A = 3 / 8, B = 6 / 8 and C = 7 / 8

        # Must appear in at least 101% --> should result in an empty output
        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 1.01)[0]), 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 1.01)[0], OrderedDict())

        # Must appear in at least 0%
        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 0)[0]), 4)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 0)[0].keys(), {"A", "B", "C", "D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 0)[0]["A"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 0)[0]["B"], 1)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 0)[0]["C"], 2)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 0)[0]["D"], 3)

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 2 / 8)[0]), 4)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 2 / 8)[0].keys(), {"A", "B", "C", "D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 2 / 8)[0]["A"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 2 / 8)[0]["B"], 1)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 2 / 8)[0]["C"], 2)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 2 / 8)[0]["D"], 3)

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 3 / 8)[0]), 4)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 3 / 8)[0].keys(), {"A", "B", "C", "D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 3 / 8)[0]["A"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 3 / 8)[0]["B"], 1)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 3 / 8)[0]["C"], 2)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 3 / 8)[0]["D"], 3)

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 4 / 8)[0]), 3)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 4 / 8)[0].keys(), {"B", "C", "D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 4 / 8)[0]["B"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 4 / 8)[0]["C"], 1)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 4 / 8)[0]["D"], 2)

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 6 / 8)[0]), 3)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 6 / 8)[0].keys(), {"B", "C", "D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 6 / 8)[0]["B"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 6 / 8)[0]["C"], 1)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 6 / 8)[0]["D"], 2)

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 7 / 8)[0]), 2)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 7 / 8)[0].keys(), {"C", "D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 7 / 8)[0]["C"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 7 / 8)[0]["D"], 1)

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 8 / 8)[0]), 1)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 8 / 8)[0].keys(), {"D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 8 / 8)[0]["D"], 0)

        # Empty input
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes({}, 0)[0], OrderedDict())
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes({}, 0.5)[0], OrderedDict())
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes({}, 1)[0], OrderedDict())
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes({}, 0)[1], OrderedDict())
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes({}, 0.5)[1], OrderedDict())
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes({}, 1)[1], OrderedDict())

    def test__count_and_filter_hashes__with_ints(self) -> None:
        d = [
            {"A": 0, "B": 0, "C": 0, "D": 0},
            {"A": 0 ,        "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {"A": 0, "B": 0, "C": 0, "D": 0},
            {                        "D": 0}
        ]  # A = 3 / 8, B = 6 / 8 and C = 7 / 8

        # Must appear in at least 1 time --> result should contain all keys
        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 1)[0]), 4)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 1)[0]["A"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 1)[0]["B"], 1)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 1)[0]["C"], 2)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 1)[0]["D"], 3)

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 2)[0]), 4)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 2)[0].keys(), {"A", "B", "C", "D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 2)[0]["A"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 2)[0]["B"], 1)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 2)[0]["C"], 2)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 2)[0]["D"], 3)

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 3)[0]), 4)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 3)[0].keys(), {"A", "B", "C", "D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 3)[0]["A"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 3)[0]["B"], 1)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 3)[0]["C"], 2)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 3)[0]["D"], 3)

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 4)[0]), 3)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 4)[0].keys(), {"B", "C", "D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 4)[0]["B"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 4)[0]["C"], 1)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 4)[0]["D"], 2)

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 7)[0]), 2)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 7)[0].keys(), {"C", "D"})
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 7)[0]["C"], 0)
        self.assertEqual(CircularFPFeaturizer._count_and_filter_hashes(d, 7)[0]["D"], 1)

    def test__count_and_filter_hashes__ints_vs_floats(self):
        d = [
            {"A": 0, "B": 0, "C": 0, "D": 0},
            {"A": 0 ,        "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {        "B": 0, "C": 0, "D": 0},
            {"A": 0, "B": 0, "C": 0, "D": 0},
            {                        "D": 0}
        ]  # A = 3 / 8, B = 6 / 8 and C = 7 / 8

        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 1)[0]), 4)
        self.assertEqual(len(CircularFPFeaturizer._count_and_filter_hashes(d, 1.0)[0]), 1)

    def test__train_with_frequent_substrset(self) -> None:
        # appears on ALL molecules
        self.assertEqual(len(CircularFPFeaturizer(only_freq_subs=True, min_subs_freq=1).fit(self.mols)), 0)

        # All data for fit and transform
        n_old = np.inf
        for freq in np.arange(0, 1.01, 0.01):
            fprinter = CircularFPFeaturizer(only_freq_subs=True, min_subs_freq=freq).fit(self.mols)

            # set of frequent pattern should get smaller when we require the patterns to be appear in more molecules
            n_new = len(fprinter)
            self.assertTrue(n_new <= n_old)
            n_old = n_new

            # Check dimension of transformed output
            fps_mat = fprinter.transform(self.mols)
            self.assertEqual(len(fprinter), fps_mat.shape[1])

            # Check frequency of substructures in the output
            freq_hash_set_inv = {v: k for k, v in fprinter.freq_hash_set_.items()}
            for j in range(len(fprinter)):
                h = freq_hash_set_inv[j]
                self.assertTrue(np.sum(fps_mat[:, j].data > 0) / self.n_mols >= fprinter.hash_cnts_filtered_[h][1])

        # Half of the data for fit and the other half for transform
        n_old = np.inf
        for freq in np.arange(0, 1.01, 0.01):
            fprinter = CircularFPFeaturizer(only_freq_subs=True, min_subs_freq=freq).fit(self.mols[:7])

            # set of frequent pattern should get smaller when we require the patterns to be appear in more molecules
            n_new = len(fprinter)
            self.assertTrue(n_new <= n_old)
            n_old = n_new

            # Check dimension of transformed output
            fps_mat = fprinter.transform(self.mols[7:])
            self.assertEqual(len(fprinter), fps_mat.shape[1])

    def test__determine_not_fitted_yet(self) -> None:
        fprinter = CircularFPFeaturizer(only_freq_subs=True)

        with self.assertRaises(NotFittedError):
             len(fprinter)
        with self.assertRaises(NotFittedError):
            fprinter.transform(self.mols)

        self.assertEqual(len(fprinter.fit(self.mols)), 86)

    def test__to_dense_output(self) -> None:
        # Output to large to be converted to a dense matrix
        fpr_mat = CircularFPFeaturizer(fp_mode="binary_folded", output_dense_matrix=True, n_bits_folded=2048,
                                       max_n_bits_for_dense_output=2048).fit_transform(self.mols)
        self.assertFalse(isspmatrix_csr(fpr_mat))
        self.assertTrue(isinstance(fpr_mat, np.ndarray))

        # Output to large to be converted to a dense matrix
        fpr_mat = CircularFPFeaturizer(fp_mode="binary_folded", output_dense_matrix=True, n_bits_folded=2048,
                                       max_n_bits_for_dense_output=100).fit_transform(self.mols)
        self.assertTrue(isspmatrix_csr(fpr_mat))
        self.assertFalse(isinstance(fpr_mat, np.ndarray))

        # Save-guard works for hashed fingerprints
        fpr_mat = CircularFPFeaturizer(output_dense_matrix=True).fit_transform(self.mols)
        self.assertTrue(isspmatrix_csr(fpr_mat))
        self.assertFalse(isinstance(fpr_mat, np.ndarray))

    def test__hashed_counting_fingerprints__ecfp(self) -> None:
        fprintr = CircularFPFeaturizer()

        fps_mat_smi = fprintr.fit_transform(self.smis)  # using SMILES
        fps_mat_mol = fprintr.fit_transform(self.mols)  # using Mol objects

        # Output shape
        self.assertEqual(fps_mat_smi.shape[0], self.n_mols)
        self.assertEqual(fps_mat_smi.shape[1], fprintr.max_hash_value_)
        self.assertEqual(fps_mat_mol.shape[0], self.n_mols)
        self.assertEqual(fps_mat_mol.shape[1], fprintr.max_hash_value_)

        # Fingerprint matrix structure
        for i, mol in enumerate(self.mols):
            fps_ref = GetMorganFingerprint(mol, radius=fprintr.radius, useFeatures=fprintr.use_features_,
                                           useChirality=fprintr.use_chirality, useCounts=fprintr.use_counts_)
            for hash, cnt in fps_ref.GetNonzeroElements().items():
                self.assertEqual(fps_mat_smi[i, hash], cnt)
                self.assertEqual(fps_mat_mol[i, hash], cnt)

    def test__hashed_counting_fingerprints__fcfp(self) -> None:
        fprintr = CircularFPFeaturizer(fp_type="FCFP")

        fps_mat_smi = fprintr.fit_transform(self.smis)  # using SMILES
        fps_mat_mol = fprintr.fit_transform(self.mols)  # using Mol objects

        # Output shape
        self.assertEqual(fps_mat_smi.shape[0], self.n_mols)
        self.assertEqual(fps_mat_smi.shape[1], fprintr.max_hash_value_)
        self.assertEqual(fps_mat_mol.shape[0], self.n_mols)
        self.assertEqual(fps_mat_mol.shape[1], fprintr.max_hash_value_)

        # Fingerprint matrix structure
        for i, mol in enumerate(self.mols):
            fps_ref = GetMorganFingerprint(mol, radius=fprintr.radius, useFeatures=fprintr.use_features_,
                                           useChirality=fprintr.use_chirality, useCounts=fprintr.use_counts_)
            for hash, cnt in fps_ref.GetNonzeroElements().items():
                self.assertEqual(fps_mat_smi[i, hash], cnt)
                self.assertEqual(fps_mat_mol[i, hash], cnt)

    def test__hashed_binary_fingerprints__ecfp(self) -> None:
        fprintr = CircularFPFeaturizer(fp_mode="binary")

        fps_mat_smi = fprintr.fit_transform(self.smis)  # using SMILES
        fps_mat_mol = fprintr.fit_transform(self.mols)  # using Mol objects

        # Output shape
        self.assertEqual(fps_mat_smi.shape[0], self.n_mols)
        self.assertEqual(fps_mat_smi.shape[1], fprintr.max_hash_value_)
        self.assertEqual(fps_mat_mol.shape[0], self.n_mols)
        self.assertEqual(fps_mat_mol.shape[1], fprintr.max_hash_value_)

        # Fingerprint matrix structure
        for i, mol in enumerate(self.mols):
            fps_ref = GetMorganFingerprint(mol, radius=fprintr.radius, useFeatures=fprintr.use_features_,
                                           useChirality=fprintr.use_chirality, useCounts=fprintr.use_counts_)
            for hash in fps_ref.GetNonzeroElements():
                self.assertTrue(fps_mat_smi[i, hash])
                self.assertTrue(fps_mat_mol[i, hash])

            # No other elements are set
            self.assertEqual(np.sum(fps_mat_smi[i, :].data), len(fps_ref.GetNonzeroElements()))
            self.assertEqual(np.sum(fps_mat_mol[i, :].data), len(fps_ref.GetNonzeroElements()))

    def test__folded_binary_fingerprints__ecfp(self) -> None:
        fprintr = CircularFPFeaturizer(fp_mode="binary_folded", n_bits_folded=512)

        fps_mat_smi = fprintr.fit_transform(self.smis)  # using SMILES
        fps_mat_mol = fprintr.fit_transform(self.mols)  # using Mol objects

        # Output shape
        self.assertEqual(fps_mat_smi.shape[0], self.n_mols)
        self.assertEqual(fps_mat_smi.shape[1], fprintr.n_bits_folded)
        self.assertEqual(fps_mat_mol.shape[0], self.n_mols)
        self.assertEqual(fps_mat_mol.shape[1], fprintr.n_bits_folded)

        # Fingerprint matrix structure
        for i, mol in enumerate(self.mols):
            fps_ref = GetMorganFingerprintAsBitVect(mol, radius=fprintr.radius, useFeatures=fprintr.use_features_,
                                                    useChirality=fprintr.use_chirality, nBits=fprintr.n_bits_folded)
            on_bits = list(fps_ref.GetOnBits())
            for j in range(fprintr.n_bits_folded):
                if j in on_bits:
                    self.assertTrue(fps_mat_smi[i, j])
                    self.assertTrue(fps_mat_mol[i, j])
                else:
                    self.assertFalse(fps_mat_smi[i, j])
                    self.assertFalse(fps_mat_mol[i, j])

    def test__rdkit_parameters_are_correct(self) -> None:
        fprintr = CircularFPFeaturizer(fp_type="ECFP", fp_mode="count").fit(None)
        self.assertTrue(fprintr.use_counts_)
        self.assertFalse(fprintr.use_features_)

        fprintr = CircularFPFeaturizer(fp_type="FCFP", fp_mode="count").fit(None)
        self.assertTrue(fprintr.use_counts_)
        self.assertTrue(fprintr.use_features_)

        fprintr = CircularFPFeaturizer(fp_type="ECFP", fp_mode="binary").fit(None)
        self.assertFalse(fprintr.use_counts_)
        self.assertFalse(fprintr.use_features_)

        fprintr = CircularFPFeaturizer(fp_type="FCFP", fp_mode="binary").fit(None)
        self.assertFalse(fprintr.use_counts_)
        self.assertTrue(fprintr.use_features_)

        fprintr = CircularFPFeaturizer(fp_type="ECFP", fp_mode="binary_folded").fit(None)
        self.assertFalse(fprintr.use_counts_)
        self.assertFalse(fprintr.use_features_)

        fprintr = CircularFPFeaturizer(fp_type="FCFP", fp_mode="binary_folded").fit(None)
        self.assertFalse(fprintr.use_counts_)
        self.assertTrue(fprintr.use_features_)

    def test__sklearn_get_params(self):
        fprinter = CircularFPFeaturizer()
        print(fprinter.get_params())

    def test__sklearn_clone(self):
        fprinter = CircularFPFeaturizer()
        _ = clone(fprinter)

    def test__parallel_fingerprinter(self):
        fps_from_sgl = \
            CircularFPFeaturizer(n_jobs=1, fp_mode="binary_folded", output_dense_matrix=True).fit_transform(self.smis)
        fps_from_par = \
            CircularFPFeaturizer(n_jobs=4, fp_mode="binary_folded", output_dense_matrix=True).fit_transform(self.smis)

        self.assertEqual((self.n_mols, 2048), fps_from_par.shape)
        np.testing.assert_equal(fps_from_sgl, fps_from_par)


if __name__ == '__main__':
    unittest.main()
