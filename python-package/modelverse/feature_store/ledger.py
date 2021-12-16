from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from schema import Or, Schema


class Ledger:
    """ A ledger stores feature metadata.

    Every feature table in a feature store is accompanied by its ledger. The ledger stores feature names and their
    properties. Any arbitrary feature properties can be added to a ledger with the exception of the following
    `reserved-properties` that exist by default in every ledger:

    - **feature_name**: Property specifying name of the feature. Also the key of the ledger.
    - **filename**: Filename where the feature is stored.
    - **description**: Feature description.
    - **tags**: A list of tags assigned to feature.

    Under the hood, a ledger is stored as a pandas dataframe with feature properties as columns and the reserved
    property `feature_name` as the primary key of the dataframe.

    Attributes:
        path (pathlib.Path): Path to directory where ledger is stored.
        data (pandas.DataFrame): Ledger dataframe.
        reserved_roperties (list): List of `reserved-properties`.

    """

    logger = logging.getLogger(__name__ + ".Ledger")

    def __init__(self, path: Path):
        """ Initiate a ledger.

        Args:
            path (pathlib.Path): Path to ledger.
        """

        self.path = path
        self.data = pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                  'filename': pd.Series(dtype=str),
                                  'description': pd.Series(dtype=str),
                                  'tags': pd.Series(dtype=object),
                                  }).reset_index(drop=True)  # reset changes index from Index() to RangeIndex()
        self.reserved_properties = ['feature_name', 'filename', 'description', 'tags']

    def save(self):
        """ Save ledger (saved as `ledger.pkl` at `self.path`). Will fail if this file already exists. """
        path = self.path / 'ledger.pkl'
        if path.exists():
            raise FileExistsError(f"File {path} already exists")
        else:
            self.data.to_pickle(path)

    def load(self) -> Ledger:
        """ Load ledger.

        Returns:
            Ledger: Returns an instance of Ledger class
        """
        self.data = pd.read_pickle(self.path / 'ledger.pkl')
        return self

    def delete(self):
        """ Delete ledger (deletes `ledger.pkl`) """
        path = self.path / 'ledger.pkl'
        path.unlink(missing_ok=True)

    def list_feature_names(self, regex=None) -> List[str]:
        """ List feature names in ledger.

        Args:
            regex (str, optional): Regex to search feature names. If None, list all feature names.

        Returns:
            list: List of matching feature names in ledger.

        """
        ret = list(self.data['feature_name'].unique())  # uniques returned in order of appearance
        if regex is not None:
            ret = list(filter(re.compile(regex).search, ret))
        return ret  # todo: think sort order

    def __write_feature_property(self, property_name: str, property_values: Dict[str, Any]):
        """ Write feature property to ledger. This function CAN be used to write SOME `reserved-properties`.

        If the property exists, existing values of specified features are overwritten by new values, otherwise a new
        property is created. If a new property is created, values of unspecified features are filled as None.

        Args:
            property_name (str): Property name to write.
            property_values (Dict[str, Any]): Dict specifying property values of features as
                {`feature name`: `property value`} pairs. An incorrect feature names raises error.

        Returns:

        """
        assert property_name not in ['feature_name', 'filename'], \
            f"'{property_name}' can not be 'feature_name' or 'filename'"
        Schema(str).validate(property_name)
        Schema(Or({str: object}, {})).validate(property_values)
        assert set(property_values.keys()).issubset(set(self.list_feature_names())), \
            f"features {set(property_values.keys()) - set(self.list_feature_names())} do not exist in ledger"

        orig_columns = copy.deepcopy(list(self.data.columns))

        if property_name in orig_columns:
            # existing property values will be overwritten by new ones
            self.logger.warning(f"Property {property_name} exists. Will overwrite new values.")
            p_values = dict(zip(self.data['feature_name'], self.data[property_name]))
            p_values.update(property_values)
            del self.data[property_name]
            new_columns = orig_columns
        else:
            self.logger.info(f"Creating new feature property {property_name}")
            p_values = property_values
            new_columns = orig_columns + [property_name]

        p_values = pd.DataFrame(p_values.items(), columns=['feature_name', property_name])

        self.data = self.data.merge(p_values, how='left', on=['feature_name'])[new_columns]

    def write_feature_property(self, property_name: str, property_values: Dict[str, Any]):
        """ Write feature property to ledger.

        If the property exists, existing values of specified features are overwritten by new values, otherwise a new
        property is created. If a new property is created, values of unspecified features are filled as None.

        This function cannot be used to write `reserved-properties`.

        Args:
            property_name (str): Property name to write.
            property_values (Dict[str, Any]): Dict specifying property values of features as
                {`feature name`: `property value`} pairs. An incorrect feature names raises error.

        Returns:

        """
        assert property_name not in self.reserved_properties, f"'{property_name}' is a reserved-property"
        self.__write_feature_property(property_name=property_name, property_values=property_values)

    def read_feature_properties(self, property_names: Union[list, str] = None,
                                feature_names: Union[list, str] = None) -> pd.DataFrame:
        """ Read feature properties from ledger.

        Reads given feature properties of given feature names.

        Args:
            property_names (Union[list, str]): List of property names to read or regex to read matching property names.
                If None, read all properties.
            feature_names (Union[list, str]): List of feature names of read ot regex to read matching feature names.
                If None, read all feature names.

        Returns:
            pd.DataFrame: Dataframe containing columns `feature_name` and property names.

        """
        Schema(Or(list, str, None)).validate(property_names)
        Schema(Or(list, str, None)).validate(feature_names)

        # None is equivalent to empty regex (search all)
        if property_names is None:
            property_names = ''
        if feature_names is None:
            feature_names = ''

        # regex to list
        if isinstance(property_names, str):
            property_names = list(filter(re.compile(property_names).search, list(self.data.columns)))
        if isinstance(feature_names, str):
            feature_names = list(filter(re.compile(feature_names).search, list(self.data['feature_name'])))

        # add mandatory 'feature_name'
        property_names = ['feature_name'] + [_ for _ in property_names if _ != 'feature_name']

        # checks
        for _ in property_names:
            assert _ in self.data.columns

        # result
        ret = self.data.loc[self.data['feature_name'].isin(feature_names), property_names].reset_index(drop=True)
        return ret

    def delete_feature_properties(self, property_names: Union[list, str] = None):
        """ Delete feature properties from ledger.

        This will not delete any `reserved-properties`.

        Args:
            property_names (Union[list, str]): List of property names to delete or regex to delete matching property
                names. If None, delete all non reserved properties.

        Returns:

        """
        # checks
        Schema(Or(list, str, None)).validate(property_names)

        # resolve property_names
        if property_names is None:
            property_names = [_ for _ in self.data.columns if _ not in self.reserved_properties]

        if isinstance(property_names, str):
            all_property_names = self.data.columns
            non_res_property_names = [_ for _ in all_property_names if _ not in self.reserved_properties]
            property_names = list(filter(re.compile(property_names).search, non_res_property_names))

        # must not delete reserved properties
        for item in self.reserved_properties:
            if item in property_names:
                raise ValueError('Cannot delete reserved-property %s .Aborting.' % item)

        # delete non-reserved properties
        for p in property_names:
            if p in self.data.columns:
                del self.data[p]
                self.logger.warning(f"Deleted property {p}")

    def write_feature_descriptions(self, descriptions: Dict[str, str]):
        """ Write feature descriptions to ledger.

        Args:
            descriptions (Dict[str, str]): Dict containing {`feature name`: `description`} pairs.

        Returns:

        """
        # descriptions is of the form {feature_name: description}
        Schema(Or({str: str}, {})).validate(descriptions)
        if (descriptions == {}):
            return
        for k, v in descriptions.items():
            assert isinstance(v, str), f"description of feature '{k}' must be string"
        self.__write_feature_property(property_name='description', property_values=descriptions)

    def read_feature_descriptions(self, feature_names: Union[list, str] = None) -> dict:
        """ Read feature descriptions from ledger.

        Args:
            feature_names (Union[list, str]): List of feature names to read descriptions for or regex to read
                descriptions for matching feature names.

        Returns:
            dict: Dict containing {`feature name`: `description`} key-value pairs.

        """
        ret = self.read_feature_properties(property_names=['description'], feature_names=feature_names)
        ret = dict(zip(ret['feature_name'], ret['description']))
        return ret

    def delete_feature_descriptions(self, feature_names: Union[list, str] = None):
        """ Delete feature descriptions from ledger.

        Args:
            feature_names (Union[list, str]): List of feature names to delete descriptions for or regex to delete
                descriptions for matching feature names. If None, deletes descriptions of all features.

        Returns:

        """
        Schema(Or(list, str, None)).validate(feature_names)
        if feature_names is None:
            feature_names = list(self.data['feature_name'])
        if isinstance(feature_names, str):
            feature_names = list(filter(re.compile(feature_names).search, list(self.data['feature_name'])))
        self.data.loc[self.data['feature_name'].isin(feature_names), 'description'] = None

    def write_feature_tags(self, tags: dict):
        """ Write feature tags to ledger.

        Given list of tags overwrite existing tags (do not append to existing tags).
        Empty string `''` feature tags are not allowed. To assign no tags to a feature use empty list
        `{feature name: []}`.

        Args:
            tags (dict): Dict containing {`feature name`: `[tags]`} pairs.

        Returns:

        """
        # tags is a dict of form {feature_name: [tag_names]}
        Schema({str: [str]}).validate(tags)

        # trivial case
        if (tags == {}) | (tags is None):
            return

        # check feature_names exist in ledger
        assert set(tags.keys()).issubset(set(self.list_feature_names())), \
            f"features '{set(tags.keys()) - set(self.list_feature_names())}' do not exist in ledger"

        # remove duplicate tags
        tags = {k: sorted(list(set(v))) for k, v in tags.items()}

        # check empty tag
        for k, v in tags.items():
            assert '' not in v, "tag can not be an empty string"

        # overwrite new tags
        orig_columns = list(self.data.columns)
        p_values = dict(zip(self.data['feature_name'], self.data['tags']))
        p_values.update(tags)
        p_values = pd.DataFrame(p_values.items(), columns=['feature_name', 'tags'])
        self.data = self.data.drop('tags', axis=1).merge(p_values, how='left', on=['feature_name'])[orig_columns]

    def read_feature_tags(self, tags: Union[list, str] = None, feature_names: Union[list, str] = None) -> dict:
        """ Read feature tags from ledger.

        Note that features with no tags are never included in the results.

        Args:
            tags (Union[list, str]): List of tags to read or regex to read matching tags. If None, read all tags.
            feature_names (Union[list, str]): List of feature names to read tags for or regex to read tags  for
                matching feature names. If None, read all feature names.

        Returns:
            dict: Dict of {`tag name`: `[feature names]`} pairs.

        """
        Schema(Or(list, str, None)).validate(tags)
        Schema(Or(list, str, None)).validate(feature_names)

        # handle None
        if feature_names is None:
            feature_names = ''

        # regex to list
        if isinstance(feature_names, str):
            feature_names = list(filter(re.compile(feature_names).search, list(self.data['feature_name'])))
        if isinstance(tags, str):
            all_tags = [item for sublist in list(self.data['tags']) for item in sublist]
            tags = list(filter(re.compile(tags).search, all_tags))

        ret = self.data.loc[self.data['feature_name'].isin(feature_names), ['feature_name', 'tags']]
        ret = ret.explode('tags')
        if tags is not None:
            ret = ret.loc[ret['tags'].isin(tags), :]
        ret = ret[~ret['tags'].isnull()]  # remove nan tags
        ret = ret.groupby('tags', dropna=False)['feature_name'].apply(list).to_dict()
        return ret

    def delete_feature_tags(self, tags: Union[list, str] = None, feature_names: Union[list, str] = None):
        """ Delete feature tags from ledger.

        Delete all matching tags of matching feature names from ledger.

        Args:
            tags (Union[list, str], optional): List of tags to delete or regex to delete matching tags.
                If None, delete all tags of matching feature_names.
            feature_names (Union[list, str], optional): List of feature names to delete tags for or regex to delete
                tags  for matching feature names. If None, delete matching tags for all feature_names.

        Returns:

        """
        Schema(Or(list, str, None)).validate(feature_names)
        Schema(Or(list, str, None)).validate(tags)

        all_tags = list(self.read_feature_tags(tags=None, feature_names=None).keys())
        if tags is None:
            tags = all_tags
        if isinstance(tags, str):
            tags = list(filter(re.compile(tags).search, all_tags))
        if feature_names is None:
            feature_names = list(self.data['feature_name'])
        if isinstance(feature_names, str):
            feature_names = list(filter(re.compile(feature_names).search, list(self.data['feature_name'])))

        sel = self.data.loc[self.data['feature_name'].isin(feature_names), 'tags']
        modified = sel.apply(lambda x: sorted(list(set(x) - set(tags))))
        self.data.loc[self.data['feature_name'].isin(feature_names), 'tags'] = modified
