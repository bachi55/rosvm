# Changelog

## Version 0.3.2

### Improved SMILES handling
- Sometimes SMILES strings cannot be parsed and converted to rdkit mol-objects because of "explicit-valence-errors"
- A new sanitization is added, trying to recover from the error and produce a valid mol-object.

## Version 0.3.1
- make featurizer compatible with GridSearchCV (only single core, i.e. ```n_jobs=1```)

## Version 0.3.0
**Feature release**
- improved convergence of the RankSVM by using line-search and early stopping criteria

## Version 0.2.0
