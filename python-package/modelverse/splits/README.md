# Splits
A split consists of multiple folds. Splits are defined in following format:
splits = {
    <split name> : <folds>,
    <split name> : <folds>,
    ...
}

# Folds
Folds are dicts containing dataset indices defined in following format:
folds = {
    <fold name> : {'train':[], 'dataset1':[], 'dataset2':[], ...},
    <fold name> : {'train':[], 'dataset1':[], 'dataset2':[], ...},
    ...
}
Here 'train', 'dataset1', 'dataset2' are dataset names. A fold must have a dataset called 'train'. All folds must
have same dataset names. Datasets are defined by an array of indices.

