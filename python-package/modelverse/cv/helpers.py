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
