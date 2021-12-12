import math
from itertools import chain

import numpy as np
from schema import Schema

from .folds import Folds
from .splits import Splits
from ..feature_store import ImageStore, NumpyFeatureStore


def create_splits_from_config(cfg):
    Schema({
        'feature_store': object,
        'entity': str,
        'splits': {str: [str]},  # each feature name in list corresponds to a train-test fold
    }).validate(cfg)

    fs = cfg['feature_store']
    entity = cfg['entity']
    ret = dict()

    # todo: add support for arbitrary downloads: train, val-1, val-2, test, ...
    train_data_code = 1
    test_data_code = 0

    print(f"Warning! Assuming train_idx are identified by value {train_data_code} and test_idx by {test_data_code} in "
          f"features {set(chain.from_iterable(list(cfg['splits'].values())))}")

    for split_name, feature_names in cfg['splits'].items():
        f, fnames = fs.load_features(entity, feature_names=feature_names, index=None)

        if isinstance(fs, NumpyFeatureStore):
            ret[split_name] = Folds({i: {
                'train': np.where(f[:, i] == train_data_code)[0],
                'test': np.where(f[:, i] == test_data_code)[0]
            } for i, _ in enumerate(fnames)}
            )

        elif isinstance(fs, ImageStore):
            ret[split_name] = Folds({i: {
                'train': np.array([img for img, attrs in f.items() if attrs[feature_name] == train_data_code]),
                'test': np.array([img for img, attrs in f.items() if attrs[feature_name] == test_data_code])
            } for i, feature_name in enumerate(fnames)}
            )

        else:
            raise NotImplementedError(f"Feature store of type '{type(fs)}' not supported!")

    return Splits(ret)


def create_ts_folds(t, train_duration, test_duration, gap_duration, shift_duration=None, skip_start_duration=None,
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
            Fold({
                0: {'train' :[0, 1, 2, 3], 'test': [5, 6, 7]},
                1: {'train' :[...], 'test': [...]},
                2: {'train' :[...], 'test': [...]},
                ...
            })
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

    ret = dict()  # index form

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

        ret[i] = {'train': np.array(train_idx, dtype=np.int64), 'test': np.array(test_idx, dtype=np.int64)}

    return Folds(ret)
