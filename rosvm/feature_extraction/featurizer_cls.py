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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import dok_matrix
from joblib.parallel import Parallel, delayed

# RDKit imports
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint, GetMorganFingerprintAsBitVect
from rdkit.Chem import MolFromSmiles
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
from rdkit.Chem.EState.Fingerprinter import FingerprintMol as EStateFingerprinter


class FeaturizerMixin(object):
    def get_length(self):
        return self.__len__()

    def __len__(self):
        check_is_fitted(self, ["n_bits_"],
                        msg="When using frequent substructure sets, the 'fit' function must be called on a set of "
                            "molecular training structures.")

        return self.n_bits_

    @staticmethod
    def sanitize_mol(mol):
        """
        :param mol: rdkit Mol object or string, if a string is provided, it is interpreted as SMILES and an RDKit Mol
            object is generated. Otherwise, the input is passed on.

        :return: rdkit Mol object
        """
        if isinstance(mol, str):
            # Convert SMILES to mol objects if strings are provided.
            smi = mol
            mol = MolFromSmiles(smi)

            if not mol:
                raise RuntimeError("SMILES could not be parsed: '%s'." % smi)

        return mol

    @staticmethod
    def _count_and_filter_hashes(dicts, min_freq) -> [OrderedDict, OrderedDict]:
        hash_cnts = OrderedDict()
        for d in dicts:
            for h in d:
                try:
                    hash_cnts[h] += 1
                except KeyError:
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


class EStateIndFeaturizer(FeaturizerMixin, BaseEstimator, TransformerMixin):
    def __int__(self):
        """
        EState indices featurizer.
        """
        pass

    def fit(self, mols=None, y=None, groups=None):
        """
        Nothing to fit here.
        """
        self.n_bits_ = 79
        return self

    def transform(self, mols):
        """
        :param mols: list of SMILES or RDKit.Mol objects, molecules for which the EState indices should be calculated.

        :return: array-like, shape = (n_mol, n_estate_idc = 79), row-matrix storing the EState indices of the provided
            molecules
        """
        if not isinstance(mols, list) and not isinstance(mols, np.ndarray):
            raise ValueError("Input must be a list of objects.")

        # Sanitize the molecule input list
        mols = [self.sanitize_mol(mol) for mol in mols]

        # Calculate the EState indices
        idc = [EStateFingerprinter(mol)[1] for mol in mols]

        # Create the output matrix
        idc_mat = np.vstack(idc)
        assert idc_mat.shape == (len(mols), self.get_length())

        return idc_mat


class CircularFPFeaturizer(FeaturizerMixin, BaseEstimator, TransformerMixin):
    def __init__(self, fp_type="ECFP", only_freq_subs=False, min_subs_freq=0.1, fp_mode="count", n_bits_folded=2048,
                 use_chirality=False, output_dense_matrix=False, max_n_bits_for_dense_output=10000, radius=2):
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
        self.fp_type = fp_type
        if self.fp_type not in ["ECFP", "FCFP"]:
            raise ValueError("Invalid fingerprint type: '%s'. Choices are 'ECFP' and 'FCFP'.")

        self.min_subs_freq = min_subs_freq
        if self.min_subs_freq < 0 or self.min_subs_freq > 1:
            raise ValueError("Sub-structure frequency invalid: '%f'. Must be from range [0, 1].")

        self.fp_mode = fp_mode
        if self.fp_mode not in ["count", "binary", "binary_folded"]:
            raise ValueError("Invalid fingerprint mode: '%s'. Choices are 'count', 'binary' and 'binary_folded'.")

        self.only_freq_subs = only_freq_subs
        self.n_bits_folded = n_bits_folded
        self.use_chirality = use_chirality
        self.output_dense_matrix = output_dense_matrix
        self.max_n_bits_for_dense_output = max_n_bits_for_dense_output
        self.radius = radius

    def _get_fingerprints(self, mols):
        if not isinstance(mols, list) and not isinstance(mols, np.ndarray):
            raise ValueError("Input must be a list of objects.")

        # Calculate the fingerprints
        fps = [self._get_fingerprint(mol) for mol in mols]
        assert len(fps) == len(mols)

        return fps

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
        mol = self.sanitize_mol(mol)

        if self.use_binary_generator_:
            fp = GetMorganFingerprintAsBitVect(mol, radius=self.radius, useChirality=self.use_chirality,
                                               useFeatures=self.use_features_, nBits=self.n_bits_folded)
        else:
            fp = GetMorganFingerprint(mol, radius=self.radius, useChirality=self.use_chirality,
                                      useFeatures=self.use_features_, useCounts=self.use_counts_)

        return fp

    def fit(self, mols, y=None, groups=None):
        """
        :param mols: list of strings or rdkit.Chem.rdchem.Mol objects, strings are interpreted as SMILES representing
            the molecules and converted into rdkit mol objects.

        :param y: Not used

        :param groups: Not used
        """
        self.use_features_ = (self.fp_type == "FCFP")
        self.use_counts_ = (self.fp_mode == "count")
        self.use_binary_generator_ = (self.fp_mode == "binary_folded")
        self.max_hash_value_ = 2 ** 32  # hash values are stored as unsigned int (32 bit), uint32

        if self.use_binary_generator_:
            self.n_bits_ = self.n_bits_folded
            self.only_freq_subs = False
        else:
            if not self.only_freq_subs:
                self.n_bits_ = self.max_hash_value_

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

        if self.output_dense_matrix and (self.n_bits_ <= self.max_n_bits_for_dense_output):
            fps_mat = fps_mat.toarray()
        else:
            fps_mat = fps_mat.tocsr()

        return fps_mat


if __name__ == "__main__":
    print(EStateIndFeaturizer().fit_transform(
        ["CC(=O)C1=CC2=C(OC(C)(C)[C@@H](O)[C@@H]2O)C=C1", "C1COC2=CC=CC=C2C1"] * 1))