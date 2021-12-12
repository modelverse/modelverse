# Terminology

## Dataset
Rows sampled from a feature table (or images sampled from an image store) form a dataset. A dataset is specified by an
array of indices (integers in case of feature table or image names in case of image store).

Example: array([1,2,3]) represents a dataset formed by sampling rows at positions 1, 2, 3 from a feature table.

## Fold
A collection of datasets is a fold. A fold is specified as key-value pairs of dataset name and dataset indices.
NOTE - A fold must always contain a dataset named "train" which is the unique dataset used to train models for
that fold. All remaining datasets in that fold are used for validation/scoring/monitoring/etc.

Examples:
{'train': array([1, 2, 3]), 'test': array([4, 5])} is a fold consiting of 2 datasets 'train' and 'test'.
{'train': array([1, 2, 3]), 'test': array([4, 5]), 'hold': array([5, 6])} is a fold consiting of 3 datasets 'train', 'test' and 'hold'.

## Split
A split is a list of folds. All folds in a split must have the same dataset names.

Example:
[ {'train': array([1, 2]), 'val': array([3, 4])}, {'train': array([3, 4]), 'val': array([1, 2])} ] is a split
consisting of 2 folds. Each fold here has two datasets named 'train' and 'val'.

