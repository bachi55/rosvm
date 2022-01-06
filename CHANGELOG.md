# Changelog

## Version 0.5.0

- clean up conda environment file and require scikit-learn (>= 0.24)
- KernelRankSVC class now supports the "pairwise" property via estimator tags

## Version 0.4.0

- add code to estimate the parameters for the platt-probabilty estimate

## Version 0.3.4
- allow specification of minimum occurrences of sub-substructures in the circular fingerprints as integer (minimum 
  number)
- add support for sparse fingerprint string outputs for the circular fingerprinter  

## Version 0.3.3

### Parallel Fingerprint Computation
- Joblib is used to compute the fingerprints of multiple molecules in parallel 
- Joblib uses the 'multiprocessing' backend to make it work
- Small performance improvements in the feature transformation function

## Version 0.3.2

### Improved SMILES handling
- Sometimes SMILES strings cannot be parsed and converted to rdkit mol-objects because of "explicit-valence-errors"
- A new sanitization is added, trying to recover from the error and produce a valid mol-object.

## Version 0.3.1
- make featurizer compatible with GridSearchCV (only single core, i.e. ```n_jobs=1```)

## Version 0.3.0

- improved convergence of the RankSVM by using line-search and early stopping criteria

## Version 0.2.0

- initial release