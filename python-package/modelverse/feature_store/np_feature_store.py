from __future__ import annotations

import gc
import logging
import pickle
from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
from schema import Or, Schema

from .feature_catalog import FeatureCatalog
from .utils import FeatureTableNotFoundError, get_random_filename
from ..utils import delete_dir


class NumpyFeatureTable:
    """ A Numpy Feature Table.

    A Numpy Feature Table consists of feature data stored as (possibly multiple) Numpy arrays along with
    table metadata and a Feature Catalog.

    The key of a Numpy Feature Table is the index of underlying Numpy arrays
    (table columns can not be set as keys).
    Numpy Feature Tables are homogenous (contain values of the same datatype).

    Under the hood, a Numpy Feature Table consists of the below files stored at `self.table_path`:

    ```
    |-- table_path/
        |-- filename1.npy # numpy array containing a subset of features
        |-- filename1.npy.meta # numpy array metadata
        | ...
        | ...
        |-- filenamek.npy
        |-- filenamek.npy.meta
        |
        |-- feature_catalog.pkl # Feature Catalog
        |-- table.meta # Table metadata
    ```

    Note that this does not support parallel CRUD api.

    TODO's:
    1) Add support for float32 arrays and other dtypes
    2) Support ACID transactions


    """

    logger = logging.getLogger(__name__ + ".NumpyFeatureTable")

    def __init__(self, table_path: Union[str, Path]):
        """ Initiate feature table.

        Args:
            table_path (str): Path to directory containing numpy arrays, table metadata and feature catalog.

        """
        self.table_path = Path(table_path).resolve()
        self.table_metadata = {'description': None,
                               'shape': None,
                               }
        self.feature_catalog = FeatureCatalog(self.table_path)

    def _write_array(self, filename: str, arr: np.ndarray, column_names: List[str]):
        """ Write a numpy array along with its metadata.

        Args:
            filename (str): Filename to write array and its metadata to (without the dot suffix).
            arr (np.ndarray): Numpy array to write.
            column_names (List[str]): List of column names in the order of columns of `arr`.

        Returns:
            None: None

        """

        # save numpy array
        np.save(self.table_path / (filename + '.npy'), arr)

        # save numpy array metadata
        metadata = {'column_names': column_names,
                    'shape': arr.shape,
                    }
        joblib.dump(metadata, self.table_path / (filename + '.npy.meta'))

    def _read_array(self, filename: str) -> (np.ndarray, dict):
        """ Read a numpy array.

        Args:
            filename (str): Filename to read from.

        Returns:
            (np.ndarray, dict): A tuple containing numpy array and its metadata.
                Metadata is a dict containing column names and array shape.

        """

        # load numpy array
        data = np.load(self.table_path / (filename + '.npy'))

        # load numpy array metadata
        metadata = joblib.load(self.table_path / (filename + '.npy.meta'))

        return data, metadata

    def _get_random_filename(self) -> str:
        """ Get a new random unused filename.

        Returns:
            str: A new filename that does not already exist in `self.table_path`.

        """
        return get_random_filename(self.table_path)

    def create(self, description: str = None) -> NumpyFeatureTable:
        """ Create feature table.

        Throws an exception if table already exists.

        Args:
            description (str): Description of the feature table.

        Returns:
            NumpyFeatureTable: Returns self.
        """

        if self.table_path.exists():
            # throw exception if feature table directory exists
            raise FileExistsError(f"Directory {self.table_path} already exists")
        else:
            # 1) else create directory and save feature table
            self.table_path.mkdir(parents=True, exist_ok=False)

            # 2) save table metadata
            self.table_metadata = {'description': description,
                                   'shape': (0, 0),
                                   }
            joblib.dump(self.table_metadata, self.table_path / 'table.meta')

            # 3) save empty feature catalog
            self.feature_catalog.save()

        return self

    def load(self) -> NumpyFeatureTable:
        """ Load feature table.

        Throws an exception if table does not exist.

        Returns:
            NumpyFeatureTable: Returns self.

        """

        if not self.table_path.exists():
            # throw exception if feature table directory does not exist
            raise FeatureTableNotFoundError(f"Directory {self.table_path} does not exist")
        else:
            # 1) check data corruption (todo)
            # 2) load table metadata
            self.table_metadata = joblib.load(self.table_path / 'table.meta')

            # 3) load feature catalog
            self.feature_catalog = FeatureCatalog(path=self.table_path).load()
        return self

    def delete(self):
        """ Delete feature table.

        Throws an exception if table does not exist.

        Returns:
            None: None
        """
        # checks
        self.load()

        # delete table_path directory and everything within
        delete_dir(self.table_path)

        # reset attrs
        self.feature_catalog = FeatureCatalog(path=self.table_path)
        self.table_metadata = {'description': None,
                               'shape': None,
                               }

    def table_shape(self):
        """ Get shape of feature table.

        Returns:
            (int, int): (num rows, num columns) of feature table.

        """
        self.load()  # checks
        return joblib.load(self.table_path / 'table.meta')['shape']

    def write_features(self, array: np.ndarray, feature_names: list, insert_index: np.ndarray):
        """ Write features to feature table.

        Existing features are overwritten at given insert_index and new features are created.


        Args:
            array (np.ndarray): Numpy array with feature data.
            feature_names (list): List of feature names of `array` in order of columns.
            insert_index (np.array): Index where to insert `array`.

        Returns:
            None: None
        """

        # checks
        self.load()
        Schema(np.ndarray).validate(array)
        Schema(Or([], [str])).validate(feature_names)
        Schema(np.ndarray).validate(insert_index)
        assert len(array.shape) == 2, "array must be a 2D array"
        assert array.shape[1] == len(feature_names), "length of 'feature_names' does not match shape of 'array'"
        assert len(set(feature_names)) == len(feature_names), "duplicate values in 'feature_names'"
        assert insert_index.dtype == 'int64', "'inser_index' must be int64"
        assert len(set(insert_index)) == len(insert_index), "'insert_index' must contain unique indices"
        assert array.shape[0] == len(insert_index), "length of 'insert_index' does not match length of 'array'"

        # trivial case
        if len(feature_names) == 0:
            return

        current_feature_names = self.feature_catalog.list_feature_names()

        # 1) existing features are overwritten at given insert_index
        update_feature_names = [name for name in feature_names if name in current_feature_names]
        if len(update_feature_names) > 0:
            update_filenames = (self.feature_catalog
                                .read_feature_properties(property_names=['filename'],
                                                         feature_names=update_feature_names)['filename']
                                .unique()
                                )
            for filename in update_filenames:
                f, metadata = self._read_array(filename)
                column_names = metadata['column_names']
                new_f = np.full((max(f.shape[0], insert_index.max() + 1), f.shape[1]), fill_value=np.nan)
                new_f[np.arange(0, f.shape[0]), :] = f
                del f
                gc.collect()
                for i, cn in enumerate(column_names):
                    for j, fn in enumerate(feature_names):
                        if cn == fn:
                            new_f[insert_index, i] = array[:, j]
                self._write_array(filename=filename, arr=new_f, column_names=column_names)
                del new_f
                gc.collect()

        # 2) new features are serialized into new files
        new_feature_names = [name for name in feature_names if name not in current_feature_names]
        new_feature_idx = [i for i, name in enumerate(feature_names) if name not in current_feature_names]
        if len(new_feature_names) > 0:
            f = np.full((insert_index.max() + 1, len(new_feature_names)), fill_value=np.nan)
            f[insert_index, :] = array[:, new_feature_idx]
            filename = self._get_random_filename()
            self._write_array(filename=filename, arr=f, column_names=new_feature_names)
            del f
            gc.collect()
            self.feature_catalog._append_features(features={name: filename for name in new_feature_names})

        # 3) update table metadata
        nrows, ncols = self.table_metadata['shape']
        self.table_metadata['shape'] = (max(nrows, insert_index.max() + 1), ncols + len(new_feature_names))
        joblib.dump(self.table_metadata, self.table_path / 'table.meta')

    def read_features(self, feature_names: Union[list, str] = None, tags: Union[list, str] = None,
                      index: np.array = None, include_untagged_features: bool = False) -> (np.ndarray, list):
        """ Read features from feature table.

        Args:
            feature_names (Union[list, str]): List or regex to read matching feature names. Use None for all features.
            tags (Union[list, str]): List of regex to read features with matching tags. Use None for all tags.
            index (np.array): Array indices for which to read features from feature table. Use None for all indices.
            include_untagged_features (bool): Whether to include untagged features or not.

        Returns:
            (np.ndarray, list): A tuple containing features array and a list of corresponding feature
                names.

        """

        # checks
        self.load()
        Schema(Or([str], str, None)).validate(feature_names)
        Schema(Or([str], str, None)).validate(tags)
        Schema(Or(np.ndarray, None)).validate(index)
        Schema(bool).validate(include_untagged_features)
        if isinstance(feature_names, list):
            assert len(feature_names) == len(set(feature_names)), "feature_names contains duplicate features"
        nrows = self.table_shape()[0]
        if index is not None:
            assert nrows >= index.max() + 1, "index out of range"

        # get list of selected feature names to read
        sel_feature_names = self.feature_catalog.read_feature_tags(tags=tags,
                                                                   feature_names=feature_names,
                                                                   include_untagged_features=include_untagged_features
                                                                   )  # returns {tag name: [feature names]}
        sel_feature_names = sorted(list(set([name for item in sel_feature_names.values() for name in item])))
        if feature_names is not None:
            sel_feature_names = [name for name in feature_names if name in sel_feature_names]  # order in original order

        # trivial case
        if len(sel_feature_names) == 0:  # trivial cases
            if index is None:
                return np.array([[]] * nrows), []
            else:
                return np.array([[]] * len(index)), []

        # get filenames to read
        sel_filenames = (self.feature_catalog
                         .read_feature_properties(property_names=['filename'],
                                                  feature_names=sel_feature_names)['filename']
                         .unique()
                         )

        # preallocate feature array
        ret = np.full((nrows, len(sel_feature_names)), fill_value=np.nan)

        # fill feature array in order of sel_feature_names
        filled = []
        for filename in sel_filenames:
            arr, metadata = self._read_array(filename)
            column_names = metadata['column_names']

            for i, cn in enumerate(column_names):
                for j, fn in enumerate(sel_feature_names):
                    if cn == fn:
                        ret[np.arange(0, arr.shape[0]), j] = arr[:, i]
                        filled.append(cn)
                        break
            del arr
            gc.collect()

        assert set(filled) == set(sel_feature_names)

        if index is not None:
            ret = ret[index, :]  # todo: optimize
            gc.collect()

        return ret, sel_feature_names

    def delete_features(self, feature_names: Union[list, str] = None, tags: Union[list, str] = None,
                        delete_untagged_features: bool = False):
        """ Delete features from feature table.

        Note that when features are deleted, the number of rows in table remains unchanged,
        only the number of columns decreases.

        Args:
            feature_names (Union[list, str]): List or regex to delete matching features. Use None for all features.
            tags (Union[list, str]): List or regex to delete features with matching tags. Use None for all tags.
            delete_untagged_features (bool): Whether to delete untagged features or not. If False, untagged features
                will not be part of deletion.

        Returns:
            None: None
        """

        # checks
        self.load()
        Schema(Or(list, str, None)).validate(feature_names)
        Schema(Or(list, str, None)).validate(tags)
        Schema(bool).validate(delete_untagged_features)

        # get list of selected feature names to delete
        sel_feature_names = self.feature_catalog.read_feature_tags(tags=tags,
                                                                   feature_names=feature_names,
                                                                   include_untagged_features=delete_untagged_features
                                                                   )  # returns {tag name: [feature names]}
        sel_feature_names = list({name for item in sel_feature_names.values() for name in item})

        # trivial case
        if sel_feature_names == []:
            return

        # get filenames to read
        sel_filenames = (self.feature_catalog
                         .read_feature_properties(property_names=['filename'],
                                                  feature_names=sel_feature_names)['filename']
                         .unique()
                         )

        for filename in sel_filenames:
            arr, metadata = self._read_array(filename)
            column_names = metadata['column_names']
            del_idx = []
            for i, cn in enumerate(column_names):
                for j, fn in enumerate(sel_feature_names):
                    if cn == fn:
                        del_idx.append(i)
            keep_idx = [i for i in range(arr.shape[1]) if i not in del_idx]
            arr = arr[:, keep_idx]
            column_names = list(np.array(column_names)[keep_idx])
            self._write_array(filename=filename, arr=arr, column_names=column_names)
            del arr
            gc.collect()

        # update feature catalog
        self.feature_catalog._delete_features(sel_feature_names)

        # update table metadata
        nrows, ncols = self.table_metadata['shape']
        ncols = ncols - len(sel_feature_names)
        assert ncols >= 0
        self.table_metadata['shape'] = (nrows, ncols)
        joblib.dump(self.table_metadata, self.table_path / 'table.meta')

    def profile(self):
        """ Profile feature table.

        Returns:

        """
        self.load()
        return


class NumpyFeatureStore:
    """ A Numpy Feature Store.

    A Numpy Feature Store is a collection of Numpy Feature Tables.

    """

    def __init__(self, path: Union[str, Path]):
        """ Initiate feature store.
        Args:
            path (Union[str, Path]):
        """
        self.path = path
        self.metadata = {'description': None}  # feature store metadata

    def create(self, description=None) -> NumpyFeatureStore:
        """ Create feature store.

        Throws an exception if feature store already exists.

        Returns:
            NumpyFeatureStore: Returns self.

        """
        if self.path.exists():
            # throw exception if feature store directory exists
            raise FileExistsError(f"Directory {self.path} already exists")
        else:
            self.path.mkdir(parents=True, exist_ok=False)

            self.metadata['description'] = description
            with open(self.path / 'store.meta', 'wb') as f:
                pickle.dump(self.metadata, f)

        return self

    def load(self) -> NumpyFeatureStore:
        """ Load feature store.

        Throws an exception if feature store does not exist.

        Returns:
            NumpyFeatureStore: Returns self.

        """
        if not self.path.exists():
            # throw exception if feature table directory does not exist
            raise Exception(f"Directory {self.path} does not exist")
        else:
            with open(self.path / 'store.meta', 'rb') as f:
                self.metadata = pickle.load(f)

        return self

    def delete(self):
        """ Delete feature store.

        Deletes feature store and all containing tables/objects.
        Throws an exception if feature store does not exist.

        Returns:
            None: None
        """
        # check feature table exists
        self.load()

        # delete table_path directory and everything within
        delete_dir(self.path)

        # reset attrs
        self.metadata = {'description': None}

    def create_table(self, table_name: str, description: str = None) -> NumpyFeatureTable:
        """ Create a feature table in feature store.

        Throws an exception if table already exists.

        Args:
            table_name (str): Feature table name.
            description (str): Description of the feature table.

        Returns:
            NumpyFeatureTable: A Numpy feature table.

        """
        self.load()
        return NumpyFeatureTable(table_path=self.path / table_name).create(description=description)

    def load_table(self, table_name: str) -> NumpyFeatureTable:
        """ Load a feature table from feature store.

        Throws an exception if table does not exist.

        Args:
            table_name (str): Feature table name.

        Returns:
            NumpyFeatureTable: A Numpy feature table.

        """
        self.load()
        return NumpyFeatureTable(table_path=self.path / table_name).load()

    def delete_table(self, table_name: str):
        """ Delete table from feature store.

        Throws an exception if table does not exist.

        Args:
            table_name (str): Feature table name.

        Returns:
            None: None

        """
        self.load()
        NumpyFeatureTable(table_path=self.path / table_name).delete()

    def list_tables(self) -> List[str]:
        """ List table names in feature store.

        Returns:
            list: List of table names in feature store.

        """
        self.load()
        return sorted([x.name for x in self.path.glob('*') if x.is_dir()])

    def summarize(self):
        """ Summarize feature store.

        Returns:

        """
        self.load()
        pass
