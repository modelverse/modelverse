import logging
import logging.config
import os
import pprint
import shutil
import subprocess
from pathlib import Path

import numpy as np


def logging_presets():
    cfg = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': '[%(asctime)s - %(levelname)s] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'colored': {
                '()': 'colorlog.ColoredFormatter',  # colored output
                # --> %(log_color)s is very important, that's what colors the line
                'format': '%(log_color)s[%(asctime)s] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'colored',
            },
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': True,
            },
        },
    }
    logging.config.dictConfig(cfg)

    logger = logging.getLogger()
    logger.info("This is info")
    logger.debug("This is debug")
    logger.warning("This is warning")
    logger.error("This is error")


def delete_dir(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError as e:
        print(e)
        pass


def create_dir(path, overwrite=False):
    """

    Args:
        path (str):
        overwrite (bool, optional):

    Returns:

    """
    logger = logging.getLogger(__name__ + ".create_path")

    if overwrite:
        logger.warning(f"Overwriting {path}!")
        delete_dir(path)
    try:
        os.makedirs(path)
    except FileExistsError as e:
        raise e


def tree(dir_path):
    """ Print directory tree """
    batcmd = f"tree {dir_path}"
    result = subprocess.check_output(batcmd, shell=True, text=True)
    print(result)


def shortpath(path, length):
    """ Shorten given path to given length """
    return str(Path(*Path(path).parts[-length:]))


def arr2_in_arr1(arr1, arr2):
    """ Find index of arr2 in arr1 if arr2 is subset of arr1.

    Args:
        arr1 (np.array):
        arr2 (np.array):

    Returns:

    """

    sort_idx = arr1.argsort()
    ret = sort_idx[np.searchsorted(arr1, arr2, sorter=sort_idx)]
    return ret


class Registry:
    """
    Inspired from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py
    """

    logger = logging.getLogger(__name__ + ".Registry")

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self.name = name
        self.obj_map = {}  # dict of dicts
        # self.allowed_object_names = ['serialize_data', 'deserialize_data', 'cv', 'serialize_model',
        # 'deserialize_model', 'predict']

    def _do_register(self, learner_name, obj_name, obj):
        # assert obj_name in self.allowed_object_names, f"Object name must be one of {self.allowed_object_names}," \
        #                                               f" found '{obj_name}'"
        if learner_name not in self.obj_map:
            self.obj_map[learner_name] = {}

        if obj_name in self.obj_map[learner_name]:
            self.logger.warning(f"'{obj_name}' already registered for learner '{learner_name}'! Overwriting.")

        self.obj_map[learner_name][obj_name] = obj

    def register(self, learner_name, obj=None):
        if obj is None:  # used as a decorator
            def deco(func_or_class):
                self._do_register(learner_name, func_or_class.__name__, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        self._do_register(learner_name, obj.__name__, obj)

    def get(self, learner_name, obj_name):
        ret = self.obj_map.get(learner_name).get(obj_name)
        if ret is None:
            raise KeyError(f"'{obj_name}' not found for learner '{learner_name}'!")
        return ret

    def __str__(self):
        return pprint.pformat(self.obj_map)
    # todo: write __repr__, __contains__, __str__ if required
