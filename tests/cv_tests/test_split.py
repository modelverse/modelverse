import numpy as np
import pytest
from modelverse.cv import Fold, Split


def test_trivial_split():
    with pytest.raises(IndexError):
        Split()
    with pytest.raises(IndexError):
        Split([])
    with pytest.raises(AttributeError):
        Split([1])


def test_mixed_split():
    # diff dataset names
    with pytest.raises(AssertionError):
        Split([Fold({'train': np.array([1])}),
               Fold({'train': np.array([1, 2]), 'val': np.array([1])}),
               ])


def test_int_split():
    # test index
    s = Split([Fold({'train': np.array([1]), 'val': np.array([5, 6])}),
               Fold({'train': np.array([1, 2]), 'val': np.array([1])}),
               ])
    np.testing.assert_array_equal(s.index, np.array([1, 2, 5, 6]))

    # test reset_index()
    np.testing.assert_equal(s.reset_index(), Split([Fold({'train': np.array([0]), 'val': np.array([2, 3])}),
                                                    Fold({'train': np.array([0, 1]), 'val': np.array([0])})]))

    # test iter()
    ret = []
    for ix, fold in s.iter():
        ret.append((ix, fold))
    np.testing.assert_equal(ret, [(0, [('train', np.array([1])), ('val', np.array([5, 6]))]),
                                  (1, [('train', np.array([1, 2])), ('val', np.array([1]))])])

    # test print
    print(s)


def test_str_split():
    # test index
    s = Split([Fold({'train': np.array(['1']), 'val': np.array(['5', '6'])}),
               Fold({'train': np.array(['1', '2']), 'val': np.array(['1'])}),
               ])
    np.testing.assert_array_equal(s.index, np.array(['1', '2', '5', '6']))

    # test reset_index()
    np.testing.assert_equal(s.reset_index(), Split([Fold({'train': np.array([0]), 'val': np.array([2, 3])}),
                                                    Fold({'train': np.array([0, 1]), 'val': np.array([0])})]))

    # test iter()
    ret = []
    for ix, fold in s.iter():
        ret.append((ix, fold))
    np.testing.assert_equal(ret, [(0, [('train', np.array(['1'])), ('val', np.array(['5', '6']))]),
                                  (1, [('train', np.array(['1', '2'])), ('val', np.array(['1']))])])

    # test print
    print(s)
