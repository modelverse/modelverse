import math

import numpy as np

from .fold import Fold
from .split import Split
from ..feature_store import ImageStore, NumpyFeatureStore


def construct_fold(feature_name: str, entity, feature_store, dataset_codenames={0: 'train', 1: 'test'}):
    f, fnames = feature_store.load_features(entity, feature_names=[feature_name], index=None)

    if isinstance(feature_store, NumpyFeatureStore):
        ret = {}
        for k, v in dataset_codenames.items():
            ret[v] = np.where(f[:, 0] == k)[0]

    elif isinstance(feature_store, ImageStore):
        ret = {}
        for k, v in dataset_codenames.items():
            ret[v] = np.array([img for img, attrs in f.items() if attrs[feature_name] == k])

    else:
        raise NotImplementedError(f"Feature store of type '{type(feature_store)}' not supported!")

    ret = Fold(ret)
    return ret


def construct_split(feature_names: list, entity, feature_store, dataset_codenames={0: 'train', 1: 'test'}):
    ret = []
    for feature_name in feature_names:
        ret.append(construct_fold(feature_store, entity, feature_name, dataset_codenames))
    ret = Split(ret)
    return ret


def create_ts_split(t, train_duration, test_duration, gap_duration, shift_duration=None, skip_start_duration=None,
                    skip_end_duration=None):
    """General function to get time series folds.

    Generates all possible folds for time series 0, 1, ..., t-1 from right to left in the following manner:
    |---skip start-----| |----train----| |----gap----| |----test----| |--skip end--|

    Args:
        t (int): Total time duration.
        train_duration (int): Train duration.
        test_duration (int): Test duration.
        gap_duration (int): Gap duration between train and test.
        shift_duration (int, optional): Shift duration. Defaults to None in which case it is taken as test_duration and
            non-overlapping folds are created. If < test_duration, overlapping folds will be created. If 0, only one
            fold will be created.
        skip_start_duration (int, optional): Duration to skip at start before constructing folds. Defaults to None in
            which no start duration will be skipped.
        skip_end_duration (int, optional): Duration to skip at end before constructing folds. Defaults to None in which
            case no end duration will be skipped.

    Returns:
            Split([ Fold({'train' :[0, 1, 2, 3], 'test': [5, 6, 7]}),
                    Fold({'train' :[...], 'test': [...]}),
                    Fold({'train' :[...], 'test': [...]}),
                    ...
                  ])
    """

    if skip_start_duration is None:
        skip_start_duration = 0

    if skip_end_duration is None:
        skip_end_duration = 0

    if shift_duration is None:
        shift_duration = test_duration  # default setting

    idx = np.array(range(t))

    # max number of folds possible
    if shift_duration == 0:
        k = 1
    elif train_duration is not None:
        k = 1 + math.floor((t - skip_start_duration - skip_end_duration -
                            train_duration - test_duration - gap_duration) / shift_duration)
    else:
        k = 1 + math.floor(
            (t - skip_start_duration - skip_end_duration - 1 - test_duration - gap_duration) / shift_duration)
    if k <= 0:
        raise Exception("No folds possible")

    ret = []  # index form

    for i in range(k):
        # Start building folds from the right most end
        # Get [train_start, train_end] [gap_start, gap_end] [test_start, test_end]
        test_end = (t - 1) - (i * shift_duration) - skip_end_duration
        test_start = test_end - test_duration + 1
        gap_end = test_start - 1
        gap_start = gap_end - gap_duration + 1
        train_end = gap_start - 1
        train_start = skip_start_duration if train_duration is None else train_end - train_duration + 1

        assert train_start >= skip_start_duration
        assert train_end >= train_start

        # Get (train_idx, test_idx)
        train_idx = np.where((idx >= train_start) & (idx <= train_end))[0]
        test_idx = np.where((idx >= test_start) & (idx <= test_end))[0]
        assert len(set(train_idx).intersection(test_idx)) == 0

        ret.append({'train': np.array(train_idx, dtype=np.int64), 'test': np.array(test_idx, dtype=np.int64)})

    return Split(ret)
