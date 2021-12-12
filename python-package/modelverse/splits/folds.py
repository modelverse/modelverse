import numpy as np
from schema import Schema

from ..utils import arr2_in_arr1


class Folds(dict):
    def __init__(self, *args, **kw):
        """
        folds = {
                <fold name> : { <dataset name>: array([indices]), .. },
                <fold name> : { <dataset name>: array([indices]), .. }
                ...
            }
        """
        # we need ordered dict so python 3.6+
        super().__init__(*args, **kw)

        self.names = list(self.keys())
        self.datasets = list(self[self.names[0]].keys())  # ordered
        self.dtype = self[self.names[0]][self.datasets[0]].dtype
        self._index = None

        # checks
        for k, v in self.items():
            assert isinstance(k, str) | isinstance(k, int), f"Fold names must be str or int"
            assert list(v.keys()) == self.datasets, f"All folds must have same dataset names"
            assert 'train' in v.keys(), f"At least one dataset name must be 'train'"
            Schema({str: lambda x: isinstance(x, np.ndarray)}).validate(v)

    def __str__(self):
        ret = ""
        for name in self.names:
            for d in self.datasets:
                ret = f"{ret}Fold: {name:<10} Dataset: {d:<10} Num points: {len(self[name][d])}\n"

        ret = f"{ret}\n{len(self.names)} folds, {len(self.datasets)} datasets, {len(self.index)} points"
        return ret

    @property
    def index(self):
        # cached property - calculated only when called first time as it can be expensive for large folds
        if self._index is None:
            ret = np.array([], dtype=self.dtype)
            for name in self.names:
                for dataset in self.datasets:
                    ret = np.append(ret, np.array(self[name][dataset], dtype=self.dtype))
            self._index = np.unique(ret)
        return self._index

    def reset_index(self):
        index = self.index
        new_folds = dict()
        for name, fold in self.items():
            new_folds[name] = dict()
            for dataset, indices in fold.items():
                new_indices = arr2_in_arr1(arr1=index, arr2=indices)
                new_folds[name][dataset] = new_indices

        return Folds(new_folds)

    def iter(self):
        for name in self.names:
            # yields <fold name>, <watchlist>
            # where <watchlist> = [(<dataset name 1> , <dataset idx 1>), (<dataset name 2> , <dataset idx 2>), .. ]
            yield name, [(d, self[name][d]) for d in self.datasets]
