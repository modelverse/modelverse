import numpy as np

from .fold import Fold
from ..utils import arr2_in_arr1


class Split(list):
    def __init__(self, *args, **kw):
        """
        split = [ Fold(), Fold(), ... ]
        """
        super().__init__(*args, **kw)
        self.dataset_names = self[0].dataset_names
        self.dtype = self[0].dtype
        self._index = None

        # checks
        for v in self:
            assert isinstance(v, Fold)
            assert list(v.keys()) == self.dataset_names, "All folds must have same dataset names"
            assert v.dtype == self.dtype, "All folds must be of same dtype"

    @property
    def index(self):
        # cached property - calculated only when called first time as it can be expensive for large folds
        if self._index is None:
            ret = np.array([], dtype=self.dtype)
            for ix in range(len(self)):
                for dataset in self.dataset_names:
                    ret = np.append(ret, np.array(self[ix][dataset], dtype=self.dtype))
            self._index = np.unique(ret)
        return self._index

    def reset_index(self):
        index = self.index
        new_split = []
        for ix, fold in enumerate(self):
            new_split.append(Fold(
                {dataset_name: arr2_in_arr1(arr1=index, arr2=indices) for dataset_name, indices in fold.items()})
            )
        return Split(new_split)

    def iter(self):
        for ix in range(len(self)):
            # yields <fold no>, <watchlist>
            # where <watchlist> = [(<dataset name 1> , <dataset idx 1>), (<dataset name 2> , <dataset idx 2>), .. ]
            yield ix, [(d, self[ix][d]) for d in self.dataset_names]

    def __str__(self):
        ret = ""
        for ix in range(len(self)):
            for d in self.dataset_names:
                ret = f"{ret}Fold: {ix:<10} Dataset: {d:<10} Num points: {len(self[ix][d])}\n"

        ret = f"{ret}\n{len(self)} folds, {len(self.dataset_names)} datasets, {len(self.index)} points"
        return ret
