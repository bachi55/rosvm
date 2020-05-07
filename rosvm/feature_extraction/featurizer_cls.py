####
#
# The MIT License (MIT)
#
# Copyright 2019, 2020 Eric Bach <eric.bach@aalto.fi>
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
import numpy as np

from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint, GetMorganFingerprintAsBitVect
from rdkit.Chem import MolFromSmiles
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import dok_matrix


class CircularFPFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, fp_type="ECFP", only_freq_subs=False, min_subs_freq=0.1, fp_mode="count", n_bits=2048,
                 use_chirality=False):
        """
        Circular Fingerprint featurizer calculates ECFP or FCFP fingerprints from molecular structures.

        :param fp_type: string, specifying whether "ECFP" or "FCFP" fingerprints should be used.
        :param only_freq_subs: boolean, indicating whether a set of frequently appearing sub-structures should be
            learned on the molecule set passed to 'fit'. If True, the transform function only returns fingerprints that
            are frequent on the training set. Otherwise, molecule set passed to 'fit' are ignored. The minimum frequency
            of a sub-structure to be used can be specified using 'min_subs_freq'.
        :param min_subs_freq: scalar, from range (0, 1) specifying the required frequency of a sub-structure appearing
            in the training set (passed to 'fit') to be included in the final fingerprint. E.g. a value of 0.1 means,
            that a sub-structure needs to appear in at least 10% of the training molecules.
        :param fp_mode: string, specifying whether "count", "binary" or "binary_folded" fingerprints should be
            calculated.
                "count": Return counts of all frequent sub-structures
                "binary": Return only a binary value for each frequent sub-structure indicating its presence or absence
                    in a given molecule
                "binary_folded": Return a fixed length binary vector (see 'n_bits') where the sub-structure hashes
                are folded to achieve a fixed length output. This option does not use the concept of frequent
                sub-structures, i.e. training molecules are ignored.
        :param n_bits: scalar, number of bits used for the fixed length binary fingerprint vector, when
            'fp_type="binary_folded"'.
        :param use_chirality: boolean, indicating whether chirality information should be added to the generated
            fingerprint
        """
        if fp_type not in ["ECFP", "FCFP"]:
            raise ValueError("Invalid fingerprint type: '%s'. Choices are 'ECFP' and 'FCFP'.")
        self.fp_type = fp_type
        self._use_features = (self.fp_type == "FCFP")

        if min_subs_freq <= 0 or min_subs_freq >= 1:
            raise ValueError("Sub-structure frequency invalid: '%f'. Must be from range (0, 1).")
        self.min_subs_freq = min_subs_freq

        if fp_mode not in ["count", "binary", "binary_folded"]:
            raise ValueError("Invalid fingerprint mode: '%s'. Choices are 'count', 'binary' and 'binary_folded'.")
        self.fp_mode = fp_mode
        self._use_counts = (self.fp_mode == "count")
        self._use_binary_generator = (self.fp_mode == "binary_folded")

        self.only_freq_subs = only_freq_subs
        self.n_bits = n_bits
        self.use_chirality = use_chirality

        self._radius = 2
        self._max_hash_value = 2 ** 32  # hash values are stored as unsigned int (32 bit), uint32

        self._freq_hash_set = None

    def _get_fingerprint(self, mol):
        """
        Calculate the fingerprint for a molecule according to the specifications.

        :param mol: rdkit.Chem.rdchem.Mol object, molecule representation

        :return:
            rdkit.DataStructs.cDataStructs.UIntSparseIntVect, if fp_mode is "count" or "binary"

                or

            rdkit.DataStructs.cDataStructs.ExplicitBitVect, if fp_mode is "binary_folded"
        """
        if isinstance(mol, str):
            # Convert SMILES to mol objects if strings are provided.
            mol = MolFromSmiles(mol)

        if self._use_binary_generator:
            fp = GetMorganFingerprintAsBitVect(mol, radius=self._radius, useChirality=self.use_chirality,
                                               useFeatures=self._use_features)
        else:
            fp = GetMorganFingerprint(mol, radius=self._radius, useChirality=self.use_chirality,
                                      useFeatures=self._use_features, useCounts=self._use_counts)

        return fp

    def _get_fingerprints(self, mols):
        if not isinstance(mols, list):
            raise ValueError("Input must be a list of objects.")

        # Calculate the fingerprints
        fps = [self._get_fingerprint(mol) for mol in mols]
        assert len(fps) == len(mols)

        return fps

    def fit(self, mols):
        """
        :param mols: list of strings or rdkit.Chem.rdchem.Mol objects, strings are interpreted as SMILES representing
            the molecules and converted into rdkit mol objects.
        """
        if not self.only_freq_subs or self.fp_mode == "binary_folded":
            # No fitting needed
            return self

        # Calculate the fingerprints
        fps = self._get_fingerprints(mols)
        assert isinstance(fps[0], UIntSparseIntVect)

        # Count all hashes
        hash_cnts = dict()
        for fp in fps:  # O(n_mols)
            for key in fp.GetNonzeroElements().keys():  # O(max_number_of_hashes) but in practice < 100 are active
                if key in hash_cnts:
                    hash_cnts[key] += 1
                else:
                    hash_cnts[key] = 1

        # Keep only frequent hashes
        n_mols = len(mols)
        self._freq_hash_set = dict()
        i = 0
        for hash, cnt in hash_cnts:
            if cnt / n_mols >= self.min_subs_freq:
                self._freq_hash_set[hash] = i
                i += 1

        return self

    def transform(self, mols):
        fps = self._get_fingerprints(mols)

        # Construct sparse matrix from fingerprints
        n_mols = len(mols)
        if self.fp_mode == "binary_folded":
            n_bits = self.n_bits
            fps_mat = dok_matrix((n_mols, n_bits), dtype="bool")
            for i, fp in enumerate(fps):
                fps_mat[i, list(fp.GetOnBits())] = True
        else:
            n_bits = len(self._freq_hash_set) if self.only_freq_subs else self._max_hash_value
            fps_mat = dok_matrix((n_mols, n_bits), dtype=np.uint16)
            for i, fp in enumerate(fps):
                for hash, cnt in fp.GetNonzeroElements().items():
                    if self.only_freq_subs:
                        if hash in self._freq_hash_set:
                            fps_mat[i, self._freq_hash_set[hash]] = cnt
                    else:
                        fps_mat[i, hash] = cnt

        fps_mat = fps_mat.tocsr()

        return fps_mat


if __name__ == "__main__":
    print(CircularFPFeaturizer(only_freq_subs=False).fit_transform(
        ["CC(=O)C1=CC2=C(OC(C)(C)[C@@H](O)[C@@H]2O)C=C1", "C1COC2=CC=CC=C2C1"] * 5000))