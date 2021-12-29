from __future__ import annotations

import copy
import logging
import pickle
from pathlib import Path
from typing import List, Union

import joblib
from schema import Or, Schema

from .feature_catalog import FeatureCatalog
from .utils import FeatureTableNotFoundError, get_random_filename
from ..utils import delete_dir


class DictFeatureTable:
    """ A Dict Feature Table.

    A Dict Feature Table consists of feature data stored as dicts (aka feature dicts) along with
    table metadata and a Feature Catalog.

    The feature dicts are of the form:
    ```
    { row_name_1: { 'f1': 1, 'f2': 2, 'f3': 3, ... }, row_name_2: { 'f1': 1, 'f3': 5, ... } }
    ```
    where `row_name_i` denote data instances/data points/index (analogous to rows in a typical feature table)
    and `fi` denote feature names (analogous to columns in a typical feature table).

    Under the hood, a Dict Feature Table consists of the below files stored at `self.table_path`:

    ```
    |-- table_path/
        |-- filename1.dict # dict containing features
        |-- filename1.dict.meta # dict metadata
        | ...
        | ...
        |-- filenamek.dict
        |-- filenamek.dict.meta
        |
        |-- feature_catalog.pkl # Feature Catalog
        |-- table.meta # Table metadata
    ```

    """

    logger = logging.getLogger(__name__ + ".DictFeatureTable")

    def __init__(self, table_path: Union[str, Path]):
        """ Initiate feature table.

        Args:
            table_path (str): Path to directory containing feature dicts, table metadata and feature catalog.

        """
        self.table_path = Path(table_path).resolve()
        self.table_metadata = {'description': None}
        self.feature_catalog = FeatureCatalog(self.table_path)

    def _write_dicts(self, filename: str, feature_dicts: dict):
        """ Write feature dicts along with its metadata.

        Args:
            filename (str): Filename to write feature dict and its metadata to (without the dot suffix).
            feature_dicts (dict): Feature dicts to write. Of the form
                `{ row_name: {'feature_name': 'feature_value', ... }, ... }`

        Returns:
            None: None

        """

        # save dicts
        joblib.dump(feature_dicts, self.table_path / (filename + '.dict'))

        # save dicts metadata
        column_names = sorted(list(set([y for x in feature_dicts.values() for y in x.keys()])))
        metadata = {'column_names': column_names,
                    'index': list(feature_dicts.keys()),
                    'shape': (len(feature_dicts), len(column_names)),
                    }
        joblib.dump(metadata, self.table_path / (filename + '.dict.meta'))

    def _read_dicts(self, filename: str) -> (dict, dict):
        """ Read feature dicts along with metadata.

        Args:
            filename (str): Filename to read from.

        Returns:
            (dict, dict): A tuple containing feature dict and its metadata.
                Metadata is a dict containing column names and feature dict shape.

        """

        # load dicts
        feature_dicts = joblib.load(self.table_path / (filename + '.dict'))

        # load dicts metadata
        metadata = joblib.load(self.table_path / (filename + '.dict.meta'))

        return feature_dicts, metadata

    def _get_random_filename(self) -> str:
        """ Get a new random unused filename.

        Returns:
            str: A new filename that does not already exist in `self.table_path`.

        """
        return get_random_filename(self.table_path)

    def create(self, description: str = None) -> DictFeatureTable:
        """ Create feature table.

        Throws an exception if table already exists.

        Args:
            description (str): Description of the feature table.

        Returns:
            DictFeatureTable: Returns self.
        """

        if self.table_path.exists():
            # throw exception if feature table directory exists
            raise FileExistsError(f"Directory {self.table_path} already exists")
        else:
            # 1) else create directory and save feature table
            self.table_path.mkdir(parents=True, exist_ok=False)

            # 2) save table metadata
            self.table_metadata['description'] = description
            joblib.dump(self.table_metadata, self.table_path / 'table.meta')

            # 3) save empty feature catalog
            self.feature_catalog.save()

        return self

    def load(self) -> DictFeatureTable:
        """ Load feature table.

        Throws an exception if table does not exist.

        Returns:
            DictFeatureTable: Returns self.

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
        # check feature table exists
        self.load()

        # delete table_path directory and everything within
        delete_dir(self.table_path)

        # reset attrs
        self.feature_catalog = FeatureCatalog(path=self.table_path)
        self.table_metadata = {'description': None}

    def table_shape(self):
        """ Get shape of feature table.

        Returns:
            (int, int): (num rows, num columns) of feature table.

        """
        self.load()  # check

        ncols = len(self.feature_catalog.list_feature_names())
        index = []
        for filename in self.feature_catalog.data['filename'].unique():
            index += joblib.load(self.table_path / (filename + '.dict.meta'))['index']
        nrows = len(set(index))
        return nrows, ncols

    def write_features(self, data: dict):
        """ Write features to feature table.

        Existing feature values are overwritten and new features are created.


        Args:
            data (dict):

        Returns:
            None: None
        """

        # checks
        self.load()
        Schema(Or({}, {object: {str: object}})).validate(data)

        feature_names = sorted(list(set([name for feature_dict in data.values() for name in feature_dict.keys()])))

        if (len(feature_names) == 0) | (data == {}):
            return

        current_feature_names = self.feature_catalog.list_feature_names()

        # 1) existing features are overwritten
        update_feature_names = [name for name in feature_names if name in current_feature_names]
        if len(update_feature_names) > 0:
            update_filenames = (self.feature_catalog
                                .read_feature_properties(property_names=['filename'],
                                                         feature_names=update_feature_names)['filename']
                                .unique()
                                )
            for filename in update_filenames:
                feature_dicts, metadata = self._read_dicts(filename)
                column_names = metadata['column_names']
                for row, fdict in data.items():
                    filtered_fdict = {k: fdict[k] for k in fdict if k in column_names}
                    if len(filtered_fdict) > 0:
                        feature_dicts.setdefault(row, filtered_fdict).update(filtered_fdict)

                self._write_dicts(filename=filename, feature_dicts=feature_dicts)

        # 2) new features are serialized into new files
        new_feature_names = [name for name in feature_names if name not in current_feature_names]
        if len(new_feature_names) > 0:
            feature_dicts = {}
            for row, fdict in data.items():
                filtered_fdict = {k: fdict[k] for k in fdict if k in new_feature_names}
                if len(filtered_fdict) > 0:
                    feature_dicts.setdefault(row, filtered_fdict).update(filtered_fdict)

            filename = self._get_random_filename()
            self._write_dicts(filename=filename, feature_dicts=feature_dicts)
            self.feature_catalog._append_features(features={name: filename for name in new_feature_names})

    def read_features(self, feature_names: Union[list, str] = None, tags: Union[list, str] = None,
                      index: list = None, include_untagged_features: bool = False) -> (dict, list):
        """ Read features from feature table.

        Args:
            feature_names (Union[list, str]): List or regex to read matching feature names. Use None for all features.
            tags (Union[list, str]): List of regex to read features with matching tags. Use None for all tags.
            index (list): Array indices for which to read features from feature table. Use None for all indices.
            include_untagged_features (bool): Whether to include untagged features or not.

        Returns:
            (dict, list): A tuple containing features array and a list of corresponding feature
                names.

        """

        # checks
        self.load()
        Schema(Or([str], str, None)).validate(feature_names)
        Schema(Or([str], str, None)).validate(tags)
        Schema(Or(list, None)).validate(index)
        Schema(bool).validate(include_untagged_features)
        if isinstance(feature_names, list):
            assert len(feature_names) == len(set(feature_names)), "feature_names contains duplicate features"

        # check index
        if index is not None:
            rem = copy.deepcopy(set(index))
            for filename in self.feature_catalog.data['filename'].unique():
                with open(self.table_path / (filename + '.dict.meta'), 'rb') as f:
                    rem = rem - set(pickle.load(f)['index'])
            assert len(rem) == 0, f"Indices {index} not in table"

        # get list of selected feature names to read
        sel_feature_names = self.feature_catalog.read_feature_tags(tags=tags,
                                                                   feature_names=feature_names,
                                                                   include_untagged_features=include_untagged_features
                                                                   )  # returns {tag name: [feature names]}
        sel_feature_names = sorted(list(set([name for item in sel_feature_names.values() for name in item])))
        if feature_names is not None:
            sel_feature_names = [name for name in feature_names if name in sel_feature_names]  # order in original order

        if sel_feature_names == []:  # trivial cases
            return {}, []

        # get filenames to read
        sel_filenames = (self.feature_catalog
                         .read_feature_properties(property_names=['filename'],
                                                  feature_names=sel_feature_names)['filename']
                         .unique()
                         )

        # fill feature dicts
        ret = {}

        for filename in sel_filenames:
            feature_dicts, metadata = self._read_dicts(filename)
            column_names = metadata['column_names']
            for row, fdict in feature_dicts.items():
                filtered_fdict = {k: fdict[k] for k in fdict if k in sel_feature_names}
                if len(filtered_fdict) > 0:
                    ret.setdefault(row, filtered_fdict).update(filtered_fdict)

        if index is not None:
            ret = {k: ret[k] for k in ret if k in index}

        return ret, sel_feature_names

    def delete_features(self, feature_names: Union[list, str] = None, tags: Union[list, str] = None,
                        delete_untagged_features: bool = False):
        """ Delete features from feature table.

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
        sel_feature_names = [name for item in sel_feature_names.values() for name in item]

        if sel_feature_names == []:  # trivial cases
            return

        # get filenames to read
        sel_filenames = (self.feature_catalog
                         .read_feature_properties(property_names=['filename'],
                                                  feature_names=sel_feature_names)['filename']
                         .unique()
                         )

        for filename in sel_filenames:
            feature_dicts, metadata = self._read_dicts(filename)
            column_names = metadata['column_names']
            rows = list(feature_dicts.keys())
            for row in rows:
                for fname in sel_feature_names:
                    if fname in feature_dicts[row]:
                        del feature_dicts[row][fname]
                if len(feature_dicts[row]) == 0:  # prune empty dicts
                    del feature_dicts[row]

            self._write_dicts(filename=filename, feature_dicts=feature_dicts)

        self.feature_catalog._delete_features(sel_feature_names)

    def profile(self):
        """ Profile feature table.

        Returns:

        """
        self.load()
        return


class DictFeatureStore:
    """ A Dict Feature Store.

    A Dict Feature Store is a collection of Dict Feature Tables.

    """

    def __init__(self, path: Union[str, Path]):
        """ Initiate feature store.
        Args:
            path (Union[str, Path]):
        """
        self.path = path
        self.metadata = {'description': None}  # feature store metadata

    def create(self, description=None) -> DictFeatureStore:
        """ Create feature store.

        Throws an exception if feature store already exists.

        Returns:
            DictFeatureStore: Returns self.

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

    def load(self) -> DictFeatureStore:
        """ Load feature store.

        Throws an exception if feature store does not exist.

        Returns:
            DictFeatureStore: Returns self.

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

    def create_table(self, table_name: str, description: str = None) -> DictFeatureTable:
        """ Create a feature table in feature store.

        Throws an exception if table already exists.

        Args:
            table_name (str): Feature table name.
            description (str): Description of the feature table.

        Returns:
            DictFeatureTable: A Dict Feature Table.

        """
        self.load()
        return DictFeatureTable(table_path=self.path / table_name).create(description=description)

    def load_table(self, table_name: str) -> DictFeatureTable:
        """ Load a feature table from feature store.

        Throws an exception if table does not exist.

        Args:
            table_name (str): Feature table name.

        Returns:
            DictFeatureTable: A Dict Feature Table.

        """
        self.load()
        return DictFeatureTable(table_path=self.path / table_name).load()

    def delete_table(self, table_name: str):
        """ Delete table from feature store.

        Throws an exception if table does not exist.

        Args:
            table_name (str): Feature table name.

        Returns:
            None: None

        """
        self.load()
        DictFeatureTable(table_path=self.path / table_name).delete()

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
