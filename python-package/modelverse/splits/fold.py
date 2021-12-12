import numpy as np


class Fold(dict):
    def __init__(self, *args, **kw):
        """
        fold = { <dataset name>: array([indices]), <dataset name>: array([indices]), ... }
        """
        super().__init__(*args, **kw)

        self.dataset_names = list(self.keys())  # todo: ordering reqd?
        self.dtype = self[self.dataset_names[0]].dtype.type
        self._index = None

        # checks
        assert self.dtype in [np.str_, np.int64], "Dataset indices must be int or str"
        assert 'train' in self.dataset_names, "At least one dataset must be named 'train'"
        for k, v in self.items():
            assert v.dtype.type == self.dtype, "Dataset indices must be of same dtype"
            assert isinstance(k, str), "Dataset names must be string"
            assert isinstance(v, np.ndarray), "Dataset indices must be arrays"

    @property
    def index(self):
        # cached property - calculated only when called first time as it can be expensive for large folds
        if self._index is None:
            ret = np.array([], dtype=self.dtype)
            for d in self.dataset_names:
                ret = np.append(ret, np.array(self[d]))
            self._index = np.unique(ret)
        return self._index

    def __str__(self):
        ret = ""
        for d in self.dataset_names:
            ret = f"{ret}Dataset: {d:<10} Num points: {len(self[d])}\n"
        return ret
