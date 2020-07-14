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
import numpy as np

from collections import OrderedDict
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint, GetMorganFingerprintAsBitVect
from rdkit.Chem import MolFromSmiles
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import dok_matrix


class CircularFPFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, fp_type="ECFP", only_freq_subs=False, min_subs_freq=0.1, fp_mode="count", n_bits_folded=2048,
                 use_chirality=False, output_dense_matrix=False, max_n_bits_for_dense_output=10000):
        """
        Circular Fingerprint featurizer calculates ECFP or FCFP fingerprints from molecular structures.

        :param fp_type: string, specifying whether "ECFP", "ecfp", "FCFP" or "fcfp" fingerprints should be used.
        :param only_freq_subs: boolean, indicating whether a set of frequently appearing sub-structures should be
            learned on the molecule set passed to 'fit'. If True, the transform function only returns fingerprints that
            are frequent on the training set. Otherwise, molecule set passed to 'fit' are ignored. The minimum frequency
            of a sub-structure to be used can be specified using 'min_subs_freq'.
        :param min_subs_freq: scalar, from range [0, 1] specifying the required frequency of a sub-structure appearing
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
        :param n_bits_folded: scalar, number of bits used for the fixed length binary fingerprint vector, when
            'fp_type="binary_folded"'.
        :param use_chirality: boolean, indicating whether chirality information should be added to the generated
            fingerprint
        :param output_dense_matrix: boolean, indicating whether the returned fingerprint matrix should be dense.
            If False, the returned matrix will be of type "scipy.sparse.csr_matrix". This parameter should be used with
            caution. When hashed fingerprints are used, e.g. counting ECFP, than the output matrix dimension is
            (n_molecules, 2 ** 32), so huge! However, when frequent sub-sets are used, than the output dimension could
            be much smaller. There is a save-guard installed, that always returns a sparse matrix if the number of
            fingerprint bits would be larger 10000 (see "max_n_bits_for_dense_output").
        :param max_n_bits_for_dense_output: scalar, maximum number of bits allowed to convert output to a dense matrix.
            This parameter is used to override "return_dense_matrix" at run-time.
        """
        fp_type = fp_type.upper()
        if fp_type not in ["ECFP", "FCFP"]:
            raise ValueError("Invalid fingerprint type: '%s'. Choices are 'ECFP' and 'FCFP'.")
        self.fp_type = fp_type
        self._use_features = (self.fp_type == "FCFP")

        if min_subs_freq < 0 or min_subs_freq > 1:
            raise ValueError("Sub-structure frequency invalid: '%f'. Must be from range [0, 1].")
        self.min_subs_freq = min_subs_freq

        if fp_mode not in ["count", "binary", "binary_folded"]:
            raise ValueError("Invalid fingerprint mode: '%s'. Choices are 'count', 'binary' and 'binary_folded'.")
        self.fp_mode = fp_mode
        self._use_counts = (self.fp_mode == "count")
        self._use_binary_generator = (self.fp_mode == "binary_folded")

        self.only_freq_subs = only_freq_subs if not self._use_binary_generator else False
        self.n_bits_folded = n_bits_folded
        self.use_chirality = use_chirality
        self.return_dense_matrix = output_dense_matrix
        self.max_n_bits_for_dense_output = max_n_bits_for_dense_output

        self._radius = 2
        self._max_hash_value = 2 ** 32  # hash values are stored as unsigned int (32 bit), uint32

        # Parameters to fit
        if self._use_binary_generator:
            self.n_bits_ = self.n_bits_folded
        else:
            if not only_freq_subs:
                self.n_bits_ = self._max_hash_value

    def get_length(self):
        return self.__len__()

    def __len__(self):
        check_is_fitted(self, ["n_bits_"],
                        msg="When using frequent substructure sets, the 'fit' function must be called on a set of "
                            "molecular training structures.")

        return self.n_bits_

    def _get_fingerprint(self, mol):
        """
        Calculate the fingerprint for a molecule according to the specifications.

        :param mol: string or rdkit.Chem.rdchem.Mol object, a string is interpreted as SMILES representing
            the molecule and converted into an rdkit mol object.

        :return:
            rdkit.DataStructs.cDataStructs.UIntSparseIntVect, if fp_mode is "count" or "binary"

                or

            rdkit.DataStructs.cDataStructs.ExplicitBitVect, if fp_mode is "binary_folded"
        """
        if isinstance(mol, str):
            # Convert SMILES to mol objects if strings are provided.
            smi = mol
            mol = MolFromSmiles(smi)
            if not mol:
                raise RuntimeError("SMILES could not be parsed: '%s'." % smi)

        if self._use_binary_generator:
            fp = GetMorganFingerprintAsBitVect(mol, radius=self._radius, useChirality=self.use_chirality,
                                               useFeatures=self._use_features, nBits=self.n_bits_folded)
        else:
            fp = GetMorganFingerprint(mol, radius=self._radius, useChirality=self.use_chirality,
                                      useFeatures=self._use_features, useCounts=self._use_counts)

        return fp

    def _get_fingerprints(self, mols):
        if not isinstance(mols, list) and not isinstance(mols, np.ndarray):
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
        if not self.only_freq_subs:
            # No fitting needed
            return self

        # Calculate the fingerprints
        fps = self._get_fingerprints(mols)
        assert isinstance(fps[0], UIntSparseIntVect)

        # Count all hashes and filter out infrequent substructures
        self.freq_hash_set_, self.hash_cnts_filtered_ = self._count_and_filter_hashes(
            [fp.GetNonzeroElements().keys() for fp in fps], self.min_subs_freq)
        self.n_bits_ = len(self.freq_hash_set_)

        return self

    def transform(self, mols):
        if self.only_freq_subs:
            check_is_fitted(self, ["n_bits_", "freq_hash_set_", "hash_cnts_filtered_"],
                            msg="When using frequent substructure sets, the 'fit' function must be called on a set of "
                                "molecular training structures.")

        # Calculate fingerprints for the molecules
        fps = self._get_fingerprints(mols)

        # Construct sparse matrix from fingerprints
        dtype = np.uint16 if self.fp_mode == "count" else np.bool
        fps_mat = dok_matrix((len(mols), self.n_bits_), dtype=dtype)
        if self.fp_mode == "binary_folded":
            for i, fp in enumerate(fps):
                fps_mat[i, list(fp.GetOnBits())] = True
        else:
            for i, fp in enumerate(fps):
                for hash, cnt in fp.GetNonzeroElements().items():
                    if self.only_freq_subs:
                        if hash in self.freq_hash_set_:
                            fps_mat[i, self.freq_hash_set_[hash]] = cnt
                    else:
                        fps_mat[i, hash] = cnt

        if self.return_dense_matrix and (self.n_bits_ <= self.max_n_bits_for_dense_output):
            fps_mat = fps_mat.toarray()
        else:
            fps_mat = fps_mat.tocsr()

        return fps_mat

    @staticmethod
    def _count_and_filter_hashes(dicts, min_freq) -> [OrderedDict, OrderedDict]:
        hash_cnts = OrderedDict()
        for d in dicts:
            for h in d:
                if h in hash_cnts:
                    hash_cnts[h] += 1
                else:
                    hash_cnts[h] = 1

        # Filter the hashes according to the required minimum frequency
        hash_cnts_filtered = OrderedDict()
        hash_idc = OrderedDict()  # stores (key, value) pairs: key=fp-hash, value=index-in-fp-matrix
        i = 0
        for h, cnt in hash_cnts.items():
            freq = cnt / len(dicts)
            if freq >= min_freq:
                hash_cnts_filtered[h] = (cnt, freq)
                hash_idc[h] = i
                i += 1

        return hash_idc, hash_cnts_filtered


if __name__ == "__main__":
    print(CircularFPFeaturizer(only_freq_subs=False).fit_transform(
        ["CC(=O)C1=CC2=C(OC(C)(C)[C@@H](O)[C@@H]2O)C=C1", "C1COC2=CC=CC=C2C1"] * 5000))