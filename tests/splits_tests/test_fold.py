import logging

import numpy as np
import pytest
from modelverse.splits import Fold

logger = logging.getLogger()


def test_fold():
    with pytest.raises(IndexError):
        _ = Fold()
    with pytest.raises(AssertionError):
        _ = Fold({'': np.array([])})
    with pytest.raises(AssertionError):
        _ = Fold({'': np.array([1, 2])})
    with pytest.raises(AssertionError):
        Fold({'train': np.array([1, 2]), 'test': np.array(['a'])})

    f = Fold({'train': np.array([1, 2]), 'test': np.array([1])})
    assert list(f.index) == [1, 2]
    assert f.dataset_names == ['train', 'test']
    assert f.dtype == np.int64
    print(f)

    f = Fold({'train': np.array(['1', '2']), 'test': np.array(['1'])})
    assert list(f.index) == ['1', '2']
    assert f.dataset_names == ['train', 'test']
    assert f.dtype == np.str_
    print(f)
