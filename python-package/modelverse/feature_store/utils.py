import os
import uuid
from pathlib import Path
from typing import Union


class FeatureTableNotFoundError(Exception):
    pass


class FeatureTableExistsError(Exception):
    pass


class FeatureStoreNotFoundError(Exception):
    pass


class FeatureStoreExistsError(Exception):
    pass


def get_random_filename(folder: Union[str, Path]) -> str:
    """ Get a new random unused filename inside folder.

    Args:
        folder (Union[str, pathlib.Path]): Directory inside which a new random filename is needed.

    Returns:
        str: A new filename that does not already exist in `folder`.

    """
    folder = Path(folder).resolve()
    used_names = [f.name.split('.')[0] for f in os.scandir(folder) if f.is_file()]  # existing filenames without ext
    used_names = used_names + [f.name for f in os.scandir(folder)]  # existing filenames with ext + folder names
    used_names = set(used_names)

    ret = str(uuid.uuid4().hex)
    while ret in used_names:
        ret = str(uuid.uuid4().hex)
    return ret
