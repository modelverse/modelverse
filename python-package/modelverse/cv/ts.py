import math

import numpy as np

from .fold import Fold
from .split import Split


def create_ts_split_points(t: int, train_duration: int, test_duration: int, gap_duration: int,
                           shift_duration: int = None, skip_start_duration: int = None, skip_end_duration: int = None):
    """ General function to create time series folds.

    Constructs all possible time-series-style folds for array `[0, 1, ..., t-1]` from right to left in the following
    manner:
    ```
    |---skip start-----| |----train----| |----gap----| |----test----| |--skip end--|
    ```

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
        List[Dict]: List of dicts of the form
        `[{'train_start':int, 'train_end':int, 'test_start':int, 'test_end':int}, ...]`.

    """

    if skip_start_duration is None:
        skip_start_duration = 0

    if skip_end_duration is None:
        skip_end_duration = 0

    if shift_duration is None:
        shift_duration = test_duration  # default setting

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

    ret = []

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

        ret.append({'train_start': train_start, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end})

    return ret


def create_ts_split(t, train_duration, test_duration, gap_duration, shift_duration=None, skip_start_duration=None,
                    skip_end_duration=None):
    """ Same as `create_ts_split_points` but returns an instance of `Split` instead.

    Returns:
        Split: A `Split` consisting of `Folds` with datasets 'train' and 'test' adn their indices.

    """
    split_points = create_ts_split_points(t, train_duration, test_duration, gap_duration, shift_duration,
                                          skip_start_duration,
                                          skip_end_duration)
    idx = np.array(range(t))

    ret = []  # index form

    for i in range(len(split_points)):
        train_idx = np.where((idx >= split_points[i]['train_start']) & (idx <= split_points[i]['train_end']))[0]
        test_idx = np.where((idx >= split_points[i]['test_start']) & (idx <= split_points[i]['test_end']))[0]
        assert len(set(train_idx).intersection(test_idx)) == 0

        ret.append(Fold({'train': np.array(train_idx, dtype=np.int64), 'test': np.array(test_idx, dtype=np.int64)}))

    return Split(ret)
