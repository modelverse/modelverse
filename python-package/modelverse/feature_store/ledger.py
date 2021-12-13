from __future__ import annotations

import copy
import logging
import re
from typing import List, Union

import pandas as pd


class Ledger:
    logger = logging.getLogger(__name__ + ".Ledger")

    def __init__(self, path):
        self.path = path
        self.data = pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                  'filename': pd.Series(dtype=str),
                                  'description': pd.Series(dtype=str),
                                  'tags': pd.Series(dtype=object),
                                  })
        self.reserved_properties = ['feature_name', 'filename', 'description', 'tags']

    def save(self):
        self.data.to_pickle(self.path)

    def load(self) -> Ledger:
        self.data = pd.read_pickle(self.path)
        return self

    def list_feature_names(self, regex=None) -> List[str]:
        ret = list(self.data['feature_name'].unique())
        if regex is not None:
            ret = list(filter(re.compile(regex).match, ret))
        return ret  # todo: think sort order

    def write_feature_property(self, property_name: str, property_values: dict):
        # todo: safeguard overwriting filename property
        assert isinstance(property_name, str)
        assert isinstance(property_values, dict)  # {feature_name: property_value}
        assert set(property_values.keys()).issubset(set(self.list_feature_names()))

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

    def delete_feature_properties(self, property_names: Union[list, str]):
        # checks
        assert property_names is not None, "property_names must be list or str"

        # convert str regex to list
        if isinstance(property_names, str):
            all_property_names = self.data.columns
            non_res_property_names = [_ for _ in all_property_names if _ not in self.reserved_properties]
            property_names = list(filter(re.compile(property_names).match, non_res_property_names))

        # must not delete reserved properties
        for item in self.reserved_properties:
            if item in property_names:
                raise Exception('Cannot delete reserved property %s .Aborting.' % item)

        # delete non-reserved properties
        for p in property_names:
            if p in self.data.columns:
                del self.data[p]
                self.logger.warning(f"Deleted property {p}")

    def read_feature_properties(self, property_names: Union[list, str] = None,
                                feature_names: Union[list, str] = None) -> pd.DataFrame:
        # None is equivalent to empty regex (search all)
        if property_names is None:
            property_names = ''
        if feature_names is None:
            feature_names = ''

        # regex to list
        if isinstance(property_names, str):
            property_names = list(filter(re.compile(property_names).match, list(self.data.columns)))
        if isinstance(feature_names, str):
            feature_names = list(filter(re.compile(feature_names).match, list(self.data['feature_name'])))

        # add mandatory 'feature_name'
        property_names = ['feature_name'] + [_ for _ in property_names if _ != 'feature_name']

        # checks
        for _ in property_names:
            assert _ in self.data.columns

        # result
        ret = self.data.loc[self.data['feature_name'].isin(feature_names), property_names]
        return ret

    def write_feature_descriptions(self, descriptions: dict):
        # descriptions is of the form {feature_name: description}
        if (descriptions == {}) | (descriptions is None):
            return
        self.write_feature_property('description', property_values=descriptions)

    def delete_feature_descriptions(self, feature_names: Union[list, str]):
        assert feature_names is not None
        if isinstance(feature_names, str):
            feature_names = list(filter(re.compile(feature_names).match, list(self.data['feature_name'])))
        self.data.loc[self.data['feature_name'].isin(feature_names), 'description'] = None

    def read_feature_descriptions(self, feature_names: Union[list, str] = None) -> dict:
        ret = self.read_feature_properties(property_names=['description'], feature_names=feature_names)
        ret = dict(zip(ret['feature_name'], ret['description']))
        return ret

    def write_feature_tags(self, tags: dict):
        # tags is a dict of form {feature_name: [tag_names]}
        # trivial case
        if (tags == {}) | (tags is None):
            return

        # check feature_names exist in ledger
        assert set(tags.keys()).issubset(set(self.list_feature_names()))

        # remove duplicate tags
        tags = {k: list(set(v)) for k, v in tags.items()}

        # check empty tag
        for k, v in tags.items():
            assert '' not in v, "tag can not be an empty string"

        # overwrite new tags
        orig_columns = list(self.data.columns)
        p_values = dict(zip(self.data['feature_name'], self.data['tags']))
        p_values.update(tags)
        p_values = pd.DataFrame(p_values.items(), columns=['feature_name', 'tags'])
        self.data = self.data.drop('tags', axis=1).merge(p_values, how='left', on=['feature_name'])[orig_columns]

    def delete_feature_tags(self, feature_names: Union[list, str]):
        assert feature_names is not None
        if isinstance(feature_names, str):
            feature_names = list(filter(re.compile(feature_names).match, list(self.data['feature_name'])))

        self.data.loc[self.data['feature_name'].isin(feature_names), 'tags'] = None  # cant use [] directly
        self.data['tags'] = self.data['tags'].fillna({i: [] for i in self.data.index})

    def read_feature_tags(self, tags: Union[list, str] = None, feature_names: Union[list, str] = None) -> dict:
        # handle None
        if feature_names is None:
            feature_names = ''

        # regex to list
        if isinstance(feature_names, str):
            feature_names = list(filter(re.compile(feature_names).match, list(self.data['feature_name'])))
        if isinstance(tags, str):
            all_tags = [item for sublist in list(self.data['tags']) for item in sublist]
            tags = list(filter(re.compile(tags).match, all_tags))

        ret = self.data.loc[self.data['feature_name'].isin(feature_names), ['feature_name', 'tags']]
        ret = ret.explode('tags')
        if tags is not None:
            ret = ret.loc[ret['tags'].isin(tags), :]
        ret['tags'] = ret['tags'].fillna('')  # replace nan tags by ''
        ret = ret.groupby('tags', dropna=False)['feature_name'].apply(list).to_dict()
        return ret
