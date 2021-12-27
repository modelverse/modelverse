import copy

import pandas as pd
import pytest
from modelverse.feature_store.feature_catalog import FeatureCatalog
from schema import SchemaError, SchemaMissingKeyError, SchemaUnexpectedTypeError


@pytest.fixture
def empty_catalog(tmp_path):
    ret = FeatureCatalog(path=tmp_path / '0')
    ret.save()
    return ret


@pytest.fixture
def example_catalog1(tmp_path):
    ret = FeatureCatalog(path=tmp_path / '1')
    ret.data = ret.data.append([{'feature_name': 'f1', 'filename': 'n1', 'description': 'd1', 'tags': ['t1']},
                                {'feature_name': 'f2', 'filename': 'n2', 'description': 'd2', 'tags': ['t2']},
                                {'feature_name': 'f3', 'filename': 'n3', 'description': None, 'tags': ['t1', 't3']},
                                {'feature_name': 'f4', 'filename': 'n4', 'description': 'd3', 'tags': []},
                                {'feature_name': 'f5', 'filename': 'n5', 'description': None, 'tags': []},
                                ], ignore_index=True)
    ret.save()
    return ret


@pytest.fixture
def example_catalog2(tmp_path):
    ret = FeatureCatalog(path=tmp_path / '2')
    ret.data = ret.data.append(
        [{'feature_name': 'f1', 'filename': 'n1', 'description': 'd1', 'tags': ['t1'], 'p1': None},
         {'feature_name': 'f2', 'filename': 'n2', 'description': 'd2', 'tags': ['t2'], 'p1': 1.0},
         ], ignore_index=True)
    ret.save()
    return ret


@pytest.fixture
def example_catalog3(tmp_path):
    ret = FeatureCatalog(path=tmp_path / '3')
    ret.data = ret.data.append(
        [{'feature_name': 'f1', 'filename': 'n1', 'description': 'd1', 'tags': ['t1'], 'p1': None},
         {'feature_name': 'f2', 'filename': 'n2', 'description': None, 'tags': ['t2'], 'p1': 1.0},
         ], ignore_index=True)
    ret.save()
    return ret


@pytest.fixture
def example_catalog4(tmp_path):
    ret = FeatureCatalog(path=tmp_path / '4')
    ret.data = ret.data.append(
        [{'feature_name': 'f1', 'filename': 'n1', 'description': 'd1', 'tags': ['t1', 't2'], 'p1': None},
         {'feature_name': 'f2', 'filename': 'n2', 'description': None, 'tags': ['t2'], 'p1': 1.0},
         {'feature_name': 'f3', 'filename': 'n3', 'description': None, 'tags': [], 'p1': 1.0},
         ], ignore_index=True)
    ret.save()
    return ret


def test_eq(request):
    catalogs = [request.getfixturevalue(catalog) for catalog in
                ['empty_catalog', 'example_catalog1', 'example_catalog2']]
    assert catalogs[0] != catalogs[1]
    assert catalogs[0] != catalogs[2]
    assert catalogs[1] != catalogs[2]
    for i in range(3):
        assert catalogs[i] == catalogs[i]


@pytest.mark.parametrize("catalog", ['empty_catalog', 'example_catalog1'])
def test_save(catalog, request):
    catalog = request.getfixturevalue(catalog)
    catalog.save()
    catalog.save()  # multiple save/overwrites

    # save inside an uncreated directory
    catalog.path = catalog.path / 'new_folder/'
    catalog.save()
    catalog.save()  # multiple save/overwrites


@pytest.mark.parametrize("catalog", ['empty_catalog', 'example_catalog1'])
def test_load_fail(catalog, request):
    # underlying .pkl file does not exist
    catalog = request.getfixturevalue(catalog)
    catalog.delete()
    with pytest.raises(FileNotFoundError):
        catalog.load()

    # underlying .pkl file is incorrect format
    pd.DataFrame({'feature_name': []}).to_pickle(catalog.path / 'feature_catalog.pkl')
    with pytest.raises(AssertionError):
        catalog.load()


@pytest.mark.parametrize("catalog", ['empty_catalog', 'example_catalog1'])
def test_load_ok(catalog, request):
    catalog = request.getfixturevalue(catalog)
    catalog.save()

    new_catalog = copy.deepcopy(catalog)
    new_catalog.load()

    assert catalog == new_catalog


@pytest.mark.parametrize("catalog", ['empty_catalog', 'example_catalog1'])
def test_delete_fail(catalog, request):
    # delete non existing catalog
    catalog = request.getfixturevalue(catalog)
    catalog.delete()
    with pytest.raises(FileNotFoundError):
        catalog.delete()


@pytest.mark.parametrize("catalog", ['empty_catalog', 'example_catalog1'])
def test_delete_ok(catalog, request):
    # save -> delete -> try load
    catalog = request.getfixturevalue(catalog)
    catalog.delete()
    catalog.save()
    catalog.delete()
    with pytest.raises(FileNotFoundError):
        catalog.load()


@pytest.mark.parametrize("catalog, feature_names, expected_error",
                         [('example_catalog1', {'f1': 'xxx', 'f9': 'xxx'}, AssertionError),
                          ])
def test_append_features_fail(catalog, feature_names, expected_error, request):
    catalog = request.getfixturevalue(catalog)
    with pytest.raises(expected_error):
        catalog._append_features(feature_names)


@pytest.mark.parametrize("catalog, feature_names, expected_data",
                         [('empty_catalog', {'f1': 'xxx'},
                           pd.DataFrame({'feature_name': pd.Series(['f1'], dtype=str),
                                         'filename': pd.Series(['xxx'], dtype=str),
                                         'description': pd.Series([None], dtype=str),
                                         'tags': pd.Series([[]], dtype=object),
                                         })),
                          ('example_catalog2', {'f3': 'xxx'},
                           pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3'], dtype=str),
                                         'filename': pd.Series(['n1', 'n2', 'xxx'], dtype=str),
                                         'description': pd.Series(['d1', 'd2', None], dtype=str),
                                         'tags': pd.Series([['t1'], ['t2'], []], dtype=object),
                                         'p1': pd.Series([None, 1, None], dtype=float),
                                         })),
                          ])
def test_append_features_ok(catalog, feature_names, expected_data, request):
    catalog = request.getfixturevalue(catalog)
    catalog._append_features(feature_names)
    pd.testing.assert_frame_equal(catalog.data, expected_data)


@pytest.mark.parametrize("catalog, feature_names, expected_data",
                         [('empty_catalog', ['f1'],
                           pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                         'filename': pd.Series(dtype=str),
                                         'description': pd.Series(dtype=str),
                                         'tags': pd.Series(dtype=object),
                                         }).reset_index(drop=True)),
                          ('empty_catalog', [],
                           pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                         'filename': pd.Series(dtype=str),
                                         'description': pd.Series(dtype=str),
                                         'tags': pd.Series(dtype=object),
                                         }).reset_index(drop=True)),
                          ('example_catalog2', [],
                           pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                         'filename': pd.Series(['n1', 'n2'], dtype=str),
                                         'description': pd.Series(['d1', 'd2'], dtype=str),
                                         'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                         'p1': pd.Series([None, 1], dtype=float),
                                         })),
                          ('example_catalog2', ['f1', 'f9'],
                           pd.DataFrame({'feature_name': pd.Series(['f2'], dtype=str),
                                         'filename': pd.Series(['n2'], dtype=str),
                                         'description': pd.Series(['d2'], dtype=str),
                                         'tags': pd.Series([['t2']], dtype=object),
                                         'p1': pd.Series([1], dtype=float),
                                         })
                           ),
                          ])
def test_delete_features_ok(catalog, feature_names, expected_data, request):
    catalog = request.getfixturevalue(catalog)
    catalog._delete_features(feature_names)
    pd.testing.assert_frame_equal(catalog.data, expected_data)


@pytest.mark.parametrize("catalog, regex, expected",
                         [
                             ('example_catalog1', None, ['f1', 'f2', 'f3', 'f4', 'f5']),
                             ('example_catalog1', '1', ['f1']),
                             ('example_catalog1', 'zz', []),
                             ('empty_catalog', 'zz', []),
                             ('empty_catalog', None, []),
                         ])
def test_list_feature_names(catalog, regex, expected, request):
    catalog = request.getfixturevalue(catalog)
    assert catalog.list_feature_names(regex=regex) == expected


@pytest.mark.parametrize("property_name, property_values, expected_error",
                         [
                             # reserved-properties
                             ('description', {}, AssertionError),
                             ('feature_name', {}, AssertionError),
                             ('filename', {}, AssertionError),
                             ('tags', {}, AssertionError),
                             # incorrect feature name
                             ('p1', {'xxx': 'v'}, AssertionError),
                             # non-string property_name
                             (420, {}, SchemaUnexpectedTypeError),
                             # non-dict property_values
                             ('p1', 'xxx', SchemaError),
                         ])
def test_write_feature_property_fail(property_name, property_values, expected_error, request):
    catalogs = ['empty_catalog', 'example_catalog1']
    for catalog in catalogs:
        catalog = request.getfixturevalue(catalog)
        with pytest.raises(expected_error):
            catalog.write_feature_property(property_name, property_values)


@pytest.mark.parametrize("property_name, property_values, expected_data",
                         [
                             # no features
                             ('p1', {}, pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                      'filename': pd.Series(dtype=str),
                                                      'description': pd.Series(dtype=str),
                                                      'tags': pd.Series(dtype=object),
                                                      'p1': pd.Series(dtype=object),
                                                      })),
                         ])
def test_write_feature_property_ok_trivial(empty_catalog, property_name, property_values, expected_data):
    empty_catalog.write_feature_property(property_name, property_values)
    pd.testing.assert_frame_equal(empty_catalog.data, expected_data)


@pytest.mark.parametrize("property_name, property_values, expected_data",
                         [
                             # no features
                             ('p1', {},
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3', 'f4', 'f5'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3', 'n4', 'n5'], dtype=str),
                                            'description': pd.Series(['d1', 'd2', None, 'd3', None], dtype=str),
                                            'tags': pd.Series([['t1'], ['t2'], ['t1', 't3'], [], []], dtype=object),
                                            'p1': pd.Series([None, None, None, None, None], dtype=object),
                                            })),
                             # int property
                             ('p1', {'f1': 1, 'f3': 2},
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3', 'f4', 'f5'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3', 'n4', 'n5'], dtype=str),
                                            'description': pd.Series(['d1', 'd2', None, 'd3', None], dtype=str),
                                            'tags': pd.Series([['t1'], ['t2'], ['t1', 't3'], [], []], dtype=object),
                                            'p1': pd.Series([1., None, 2., None, None], dtype=float),
                                            })),
                             # str property
                             ('p1', {'f1': '1', 'f3': '2'},
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3', 'f4', 'f5'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3', 'n4', 'n5'], dtype=str),
                                            'description': pd.Series(['d1', 'd2', None, 'd3', None], dtype=str),
                                            'tags': pd.Series([['t1'], ['t2'], ['t1', 't3'], [], []], dtype=object),
                                            'p1': pd.Series(['1', None, '2', None, None], dtype=object),
                                            })),
                             # mixed property
                             ('p1', {'f1': 1, 'f3': '2'},
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3', 'f4', 'f5'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3', 'n4', 'n5'], dtype=str),
                                            'description': pd.Series(['d1', 'd2', None, 'd3', None], dtype=str),
                                            'tags': pd.Series([['t1'], ['t2'], ['t1', 't3'], [], []], dtype=object),
                                            'p1': pd.Series([1, None, '2', None, None], dtype=object),
                                            })),
                         ])
def test_write_feature_property_ok_nontrivial(example_catalog1, property_name, property_values, expected_data):
    example_catalog1.write_feature_property(property_name, property_values)
    pd.testing.assert_frame_equal(example_catalog1.data, expected_data)


@pytest.mark.parametrize("property_names, feature_names, expected_error",
                         [
                             # incorrect properties
                             (['xxx'], None, AssertionError),
                             # dtypes
                             (1, None, SchemaError),
                             (None, 1, SchemaError),
                         ])
def test_read_feature_properties_fail(property_names, feature_names, expected_error, request):
    catalogs = ['empty_catalog', 'example_catalog1']
    for catalog in catalogs:
        catalog = request.getfixturevalue(catalog)
        with pytest.raises(expected_error):
            catalog.read_feature_properties(property_names, feature_names)


@pytest.mark.parametrize("property_names, feature_names, expected_data",
                         [
                             # all properties, all features
                             (None, None,
                              pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                            'filename': pd.Series(dtype=str),
                                            'description': pd.Series(dtype=str),
                                            'tags': pd.Series(dtype=str),
                                            }).reset_index(drop=True)),
                             # selected features by list
                             (None, ['xxx'],
                              pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                            'filename': pd.Series(dtype=str),
                                            'description': pd.Series(dtype=str),
                                            'tags': pd.Series(dtype=str),
                                            }).reset_index(drop=True)),
                             # selected features by regex
                             (None, 'xxx',
                              pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                            'filename': pd.Series(dtype=str),
                                            'description': pd.Series(dtype=str),
                                            'tags': pd.Series(dtype=str),
                                            }).reset_index(drop=True)),
                             # selected properties by list
                             (['description'], None,
                              pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                            'description': pd.Series(dtype=str),
                                            }).reset_index(drop=True)),
                             # selected properties by regex
                             ('desc', None,
                              pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                            'description': pd.Series(dtype=str),
                                            }).reset_index(drop=True)),
                             ('xxx', None,
                              pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                            }).reset_index(drop=True)),
                             # reserved properties
                             (['feature_name', 'filename', 'description', 'tags'], None,
                              pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                            'filename': pd.Series(dtype=str),
                                            'description': pd.Series(dtype=str),
                                            'tags': pd.Series(dtype=str),
                                            }).reset_index(drop=True)),
                             # no properties
                             ([], None, pd.DataFrame({'feature_name': pd.Series(dtype=str)}).reset_index(drop=True)),
                             # no features
                             (None, [], pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                      'filename': pd.Series(dtype=str),
                                                      'description': pd.Series(dtype=str),
                                                      'tags': pd.Series(dtype=str),
                                                      }).reset_index(drop=True))
                         ])
def test_read_feature_properties_ok_trivial(empty_catalog, property_names, feature_names, expected_data):
    ret = empty_catalog.read_feature_properties(property_names, feature_names)
    pd.testing.assert_frame_equal(ret, expected_data)


@pytest.mark.parametrize("property_names, feature_names, expected_data",
                         [
                             # all properties, all features
                             (None, None,
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2'], dtype=str),
                                            'description': pd.Series(['d1', 'd2'], dtype=str),
                                            'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                            'p1': pd.Series([None, 1], dtype=float),
                                            })),
                             # selected features by list
                             (None, ['f2', 'xxx'],
                              pd.DataFrame({'feature_name': pd.Series(['f2'], dtype=object),
                                            'filename': pd.Series(['n2'], dtype=object),
                                            'description': pd.Series(['d2'], dtype=object),
                                            'tags': pd.Series([['t2']], dtype=object),
                                            'p1': pd.Series([1.], dtype=float),
                                            })),
                             # selected features by regex
                             (None, '2',
                              pd.DataFrame({'feature_name': pd.Series(['f2'], dtype=object),
                                            'filename': pd.Series(['n2'], dtype=object),
                                            'description': pd.Series(['d2'], dtype=object),
                                            'tags': pd.Series([['t2']], dtype=object),
                                            'p1': pd.Series([1.], dtype=float),
                                            })),
                             # selected properties by list
                             (['p1'], ['f2'],
                              pd.DataFrame({'feature_name': pd.Series(['f2'], dtype=str),
                                            'p1': pd.Series([1.], dtype=float),
                                            })),
                             # selected properties by regex
                             ('p', ['f2'],
                              pd.DataFrame({'feature_name': pd.Series(['f2'], dtype=str),
                                            'description': pd.Series(['d2'], dtype=object),
                                            'p1': pd.Series([1.], dtype=float),
                                            })),
                             ('xxx', ['f2'],
                              pd.DataFrame({'feature_name': pd.Series(['f2'], dtype=str),
                                            })),
                             # reserved properties
                             (['feature_name', 'filename', 'description', 'tags'], ['f2'],
                              pd.DataFrame({'feature_name': pd.Series(['f2'], dtype=object),
                                            'filename': pd.Series(['n2'], dtype=object),
                                            'description': pd.Series(['d2'], dtype=object),
                                            'tags': pd.Series([['t2']], dtype=object),
                                            })),
                             # no properties
                             ([], ['f2'], pd.DataFrame({'feature_name': pd.Series(['f2'], dtype=str)})),
                             # no features
                             (None, [], pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                      'filename': pd.Series(dtype=str),
                                                      'description': pd.Series(dtype=str),
                                                      'tags': pd.Series(dtype=object),
                                                      'p1': pd.Series(dtype=float),
                                                      }).reset_index(drop=True))
                         ])
def test_read_feature_properties_ok_nontrivial(example_catalog2, property_names, feature_names, expected_data):
    ret = example_catalog2.read_feature_properties(property_names, feature_names)
    pd.testing.assert_frame_equal(ret, expected_data)


@pytest.mark.parametrize("property_names, expected_error",
                         [
                             # incorrect datatype of property_names
                             ({}, SchemaError),
                             (1, SchemaError),
                             # reserved-properties
                             (['feature_name'], ValueError),
                             (['filename'], ValueError),
                             (['description'], ValueError),
                             (['tags'], ValueError),
                             (['tags', 'xxx'], ValueError),
                         ])
def test_delete_feature_properties_fail(property_names, expected_error, request):
    catalogs = ['empty_catalog', 'example_catalog1']
    for catalog in catalogs:
        catalog = request.getfixturevalue(catalog)
        with pytest.raises(expected_error):
            catalog.delete_feature_properties(property_names)


@pytest.mark.parametrize("property_names, expected_data",
                         [
                             # empty list
                             ([], pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                'filename': pd.Series(dtype=str),
                                                'description': pd.Series(dtype=str),
                                                'tags': pd.Series(dtype=object),
                                                }).reset_index(drop=True)),
                             # list
                             (['xxx'], pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                     'filename': pd.Series(dtype=str),
                                                     'description': pd.Series(dtype=str),
                                                     'tags': pd.Series(dtype=object),
                                                     }).reset_index(drop=True)),
                             # regex
                             ('xxx', pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                   'filename': pd.Series(dtype=str),
                                                   'description': pd.Series(dtype=str),
                                                   'tags': pd.Series(dtype=object),
                                                   }).reset_index(drop=True)),
                             # regex with reserved-property match
                             ('description', pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                           'filename': pd.Series(dtype=str),
                                                           'description': pd.Series(dtype=str),
                                                           'tags': pd.Series(dtype=object),
                                                           }).reset_index(drop=True)),
                             # None
                             (None, pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                  'filename': pd.Series(dtype=str),
                                                  'description': pd.Series(dtype=str),
                                                  'tags': pd.Series(dtype=object),
                                                  }).reset_index(drop=True)),
                         ])
def test_delete_feature_properties_ok_trivial(empty_catalog, property_names, expected_data):
    empty_catalog.delete_feature_properties(property_names)
    pd.testing.assert_frame_equal(empty_catalog.data, expected_data)


@pytest.mark.parametrize("property_names, expected_data",
                         [
                             # empty list
                             ([], pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                'description': pd.Series(['d1', 'd2'], dtype=str),
                                                'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                'p1': pd.Series([None, 1.], dtype=float),
                                                })),
                             # list
                             (['p1', 'xxx'], pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                           'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                           'description': pd.Series(['d1', 'd2'], dtype=str),
                                                           'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                           })),
                             # regex
                             ('xxx', pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                   'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                   'description': pd.Series(['d1', 'd2'], dtype=str),
                                                   'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                   'p1': pd.Series([None, 1.], dtype=float),
                                                   })),
                             # regex with reserved-property match
                             ('description', pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                           'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                           'description': pd.Series(['d1', 'd2'], dtype=str),
                                                           'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                           'p1': pd.Series([None, 1.], dtype=float),
                                                           })),
                             # regex with reserved-property match
                             (None, pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                  'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                  'description': pd.Series(['d1', 'd2'], dtype=str),
                                                  'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                  })),
                         ])
def test_delete_feature_properties_ok_nontrivial(example_catalog2, property_names, expected_data):
    example_catalog2.delete_feature_properties(property_names)
    pd.testing.assert_frame_equal(example_catalog2.data, expected_data)


@pytest.mark.parametrize("descriptions, expected_error",
                         [
                             # not dict
                             (1, SchemaError),
                             # None
                             (None, SchemaError),
                             # incorrect feature
                             ('xxx', SchemaError),
                             # non string descriptions
                             ({'f1': 1}, SchemaError)
                         ])
def test_write_feature_descriptions_fail(descriptions, expected_error, request):
    catalogs = ['empty_catalog', 'example_catalog1']
    for catalog in catalogs:
        catalog = request.getfixturevalue(catalog)
        with pytest.raises(expected_error):
            catalog.write_feature_descriptions(descriptions)


@pytest.mark.parametrize("descriptions, expected_data",
                         [
                             # {}
                             ({}, pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                'filename': pd.Series(dtype=str),
                                                'description': pd.Series(dtype=str),
                                                'tags': pd.Series(dtype=object),
                                                }).reset_index(drop=True))
                         ])
def test_write_feature_descriptions_ok_trivial(empty_catalog, descriptions, expected_data):
    empty_catalog.write_feature_descriptions(descriptions)
    pd.testing.assert_frame_equal(empty_catalog.data, expected_data)


@pytest.mark.parametrize("descriptions, expected_data",
                         [
                             # {}
                             ({}, pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                'description': pd.Series(['d1', None], dtype=str),
                                                'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                'p1': pd.Series([None, 1.], dtype=float),
                                                })),
                             # replace existing description
                             ({'f1': 'xxx'}, pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                           'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                           'description': pd.Series(['xxx', None], dtype=str),
                                                           'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                           'p1': pd.Series([None, 1.], dtype=float),
                                                           })),
                             # fill empty description
                             ({'f2': 'xxx'}, pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                           'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                           'description': pd.Series(['d1', 'xxx'], dtype=str),
                                                           'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                           'p1': pd.Series([None, 1.], dtype=float),
                                                           })),
                         ])
def test_write_feature_descriptions_ok_nontrivial(example_catalog3, descriptions, expected_data):
    example_catalog3.write_feature_descriptions(descriptions)
    pd.testing.assert_frame_equal(example_catalog3.data, expected_data)


@pytest.mark.parametrize("feature_names, expected_error",
                         [
                             (1, SchemaError),
                         ])
def test_read_feature_descriptions_fail(feature_names, expected_error, request):
    catalogs = ['empty_catalog', 'example_catalog1']
    for catalog in catalogs:
        catalog = request.getfixturevalue(catalog)
        with pytest.raises(expected_error):
            catalog.read_feature_descriptions(feature_names)


@pytest.mark.parametrize("feature_names, expected_data",
                         [
                             # None
                             (None, {}),
                             # list
                             ([], {}),
                             (['f1'], {}),
                             # regex
                             ('', {}),
                             ('f', {}),
                         ])
def test_read_feature_descriptions_ok_trivial(empty_catalog, feature_names, expected_data):
    ret = empty_catalog.read_feature_descriptions(feature_names)
    assert ret == expected_data


@pytest.mark.parametrize("feature_names, expected_data",
                         [
                             # None
                             (None, {'f1': 'd1', 'f2': None}),
                             # list
                             ([], {}),
                             (['f1', 'xxx'], {'f1': 'd1'}),
                             # regex
                             ('2', {'f2': None}),
                             ('xxx', {})
                         ])
def test_read_feature_descriptions_ok_nontrivial(example_catalog3, feature_names, expected_data):
    ret = example_catalog3.read_feature_descriptions(feature_names)
    assert ret == expected_data


@pytest.mark.parametrize("feature_names, expected_error",
                         [
                             (1, SchemaError),
                         ])
def test_delete_feature_descriptions_fail(feature_names, expected_error, request):
    catalogs = ['empty_catalog', 'example_catalog1']
    for catalog in catalogs:
        catalog = request.getfixturevalue(catalog)
        with pytest.raises(expected_error):
            catalog.delete_feature_descriptions(feature_names)


@pytest.mark.parametrize("feature_names, expected_data",
                         [
                             # list
                             ([], pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                'filename': pd.Series(dtype=str),
                                                'description': pd.Series(dtype=str),
                                                'tags': pd.Series(dtype=object),
                                                }).reset_index(drop=True)),
                             (['xxx'], pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                     'filename': pd.Series(dtype=str),
                                                     'description': pd.Series(dtype=str),
                                                     'tags': pd.Series(dtype=object),
                                                     }).reset_index(drop=True)),
                             # regex
                             ('xxx', pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                   'filename': pd.Series(dtype=str),
                                                   'description': pd.Series(dtype=str),
                                                   'tags': pd.Series(dtype=object),
                                                   }).reset_index(drop=True)),
                             # None
                             (None, pd.DataFrame({'feature_name': pd.Series(dtype=str),
                                                  'filename': pd.Series(dtype=str),
                                                  'description': pd.Series(dtype=str),
                                                  'tags': pd.Series(dtype=object),
                                                  }).reset_index(drop=True)),
                         ])
def test_delete_feature_descriptions_ok_trivial(empty_catalog, feature_names, expected_data):
    empty_catalog.delete_feature_descriptions(feature_names)
    pd.testing.assert_frame_equal(empty_catalog.data, expected_data)


@pytest.mark.parametrize("feature_names, expected_data",
                         [
                             # list
                             ([], pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                'description': pd.Series(['d1', None], dtype=str),
                                                'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                'p1': pd.Series([None, 1.], dtype=float),
                                                })),
                             (['f1', 'xxx'], pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                           'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                           'description': pd.Series([None, None], dtype=str),
                                                           'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                           'p1': pd.Series([None, 1.], dtype=float),
                                                           })),
                             # regex
                             ('xxx', pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                   'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                   'description': pd.Series(['d1', None], dtype=str),
                                                   'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                   'p1': pd.Series([None, 1.], dtype=float),
                                                   })),
                             ('1', pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                 'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                 'description': pd.Series([None, None], dtype=str),
                                                 'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                 'p1': pd.Series([None, 1.], dtype=float),
                                                 })),
                             # None
                             (None, pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                                  'filename': pd.Series(['n1', 'n2'], dtype=str),
                                                  'description': pd.Series([None, None], dtype=str),
                                                  'tags': pd.Series([['t1'], ['t2']], dtype=object),
                                                  'p1': pd.Series([None, 1.], dtype=float),
                                                  })),
                         ])
def test_delete_feature_descriptions_ok_nontrivial(example_catalog3, feature_names, expected_data):
    example_catalog3.delete_feature_descriptions(feature_names)
    pd.testing.assert_frame_equal(example_catalog3.data, expected_data)


@pytest.mark.parametrize("tags, expected_error",
                         [
                             # incorrect dtypes
                             (1, SchemaUnexpectedTypeError),
                             ({}, SchemaMissingKeyError),
                             ({0: []}, SchemaMissingKeyError),
                             ({'f1': [1]}, SchemaError),
                             ({'f1': [None]}, SchemaError),
                             ({'f1': 1}, SchemaError),
                             # incorrect features
                             ({'xxx': []}, AssertionError),
                         ])
def test_write_feature_tags_fail(empty_catalog, example_catalog3, tags, expected_error):
    with pytest.raises(expected_error):
        empty_catalog.write_feature_tags(tags)

    with pytest.raises(expected_error):
        example_catalog3.write_feature_tags(tags)

    with pytest.raises(AssertionError):
        example_catalog3.write_feature_tags({'f1': ['', 't1']})  # reserved tags


@pytest.mark.parametrize("tags, expected_data",
                         [
                             ({'f1': [], 'f2': ['t3', 't3', 't1']},
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2'], dtype=str),
                                            'description': pd.Series(['d1', None], dtype=str),
                                            'tags': pd.Series([[], ['t1', 't3']], dtype=object),
                                            'p1': pd.Series([None, 1.], dtype=float),
                                            })),
                         ])
def test_write_feature_tags_ok(example_catalog3, tags, expected_data):
    example_catalog3.write_feature_tags(tags)
    pd.testing.assert_frame_equal(example_catalog3.data, expected_data)


@pytest.mark.parametrize("tags, feature_names, expected_error",
                         [
                             # incorrect dtypes
                             (1, None, SchemaError),
                             (None, 1, SchemaError),
                         ])
def test_read_feature_tags_fail(tags, feature_names, expected_error, request):
    catalogs = ['empty_catalog', 'example_catalog1']
    for catalog in catalogs:
        catalog = request.getfixturevalue(catalog)
        with pytest.raises(expected_error):
            catalog.read_feature_tags(tags, feature_names)


@pytest.mark.parametrize("tags, feature_names, include_untagged_features, expected_data",
                         [
                             # list of tags
                             ([], None, False, {}),
                             ([], None, True, {}),
                             (['t1'], None, False, {}),
                             (['t1'], None, True, {}),
                             # tags regex
                             ('xxx', None, False, {}),
                             ('xxx', None, True, {}),
                             # list fo features
                             (None, [], False, {}),
                             (None, [], True, {}),
                             (None, ['xxx'], False, {}),
                             (None, ['xxx'], True, {}),
                             # features regex
                             (None, 'xxx', False, {}),
                             (None, 'xxx', True, {}),
                         ])
def test_read_feature_tags_ok_trivial(empty_catalog, tags, feature_names, include_untagged_features, expected_data):
    ret = empty_catalog.read_feature_tags(tags, feature_names, include_untagged_features)
    assert ret == expected_data


@pytest.mark.parametrize("tags, feature_names, include_untagged_features, expected_data",
                         [
                             # list of tags
                             ([], None, False, {}),
                             ([], None, True, {'': ['f3']}),
                             (['t2'], None, False, {'t2': ['f1', 'f2']}),
                             (['t2'], None, True, {'': ['f3'], 't2': ['f1', 'f2']}),
                             (['t2', 't1'], None, False, {'t1': ['f1'], 't2': ['f1', 'f2']}),
                             (['t2', 't1'], None, True, {'': ['f3'], 't1': ['f1'], 't2': ['f1', 'f2']}),
                             # tags regex
                             ('xxx', None, False, {}),
                             ('xxx', None, True, {'': ['f3']}),
                             ('t', None, False, {'t1': ['f1'], 't2': ['f1', 'f2']}),
                             ('t', None, True, {'': ['f3'], 't1': ['f1'], 't2': ['f1', 'f2']}),
                             ('', None, False, {'t1': ['f1'], 't2': ['f1', 'f2']}),
                             ('', None, True, {'': ['f3'], 't1': ['f1'], 't2': ['f1', 'f2']}),
                             ('2', None, False, {'t2': ['f1', 'f2']}),
                             ('2', None, True, {'': ['f3'], 't2': ['f1', 'f2']}),
                             # list of features
                             (None, [], False, {}),
                             (None, [], True, {}),
                             (None, ['xxx'], False, {}),
                             (None, ['xxx'], True, {}),
                             (None, ['f1'], False, {'t1': ['f1'], 't2': ['f1']}),
                             (None, ['f1'], True, {'t1': ['f1'], 't2': ['f1']}),
                             # features regex
                             (None, 'xxx', False, {}),
                             (None, 'xxx', True, {}),
                             (None, '1', False, {'t1': ['f1'], 't2': ['f1']}),
                             (None, '1', True, {'t1': ['f1'], 't2': ['f1']}),
                             # both tags and feature names are not None
                             (['t1'], '1', False, {'t1': ['f1']}),
                             (['t1'], '1', True, {'t1': ['f1']}),
                             (None, None, False, {'t1': ['f1'], 't2': ['f1', 'f2']}),
                             (None, None, True, {'t1': ['f1'], 't2': ['f1', 'f2'], '': ['f3']}),
                         ])
def test_read_feature_tags_ok_nontrivial(example_catalog4, tags, feature_names, include_untagged_features,
                                         expected_data):
    ret = example_catalog4.read_feature_tags(tags, feature_names, include_untagged_features)
    assert ret == expected_data


@pytest.mark.parametrize("tags, feature_names, expected_error",
                         [
                             # incorrect dtypes
                             (1, None, SchemaError),
                             (None, 1, SchemaError),
                         ])
def test_delete_feature_tags_fail(tags, feature_names, expected_error, request):
    catalogs = ['empty_catalog', 'example_catalog1']
    for catalog in catalogs:
        catalog = request.getfixturevalue(catalog)
        with pytest.raises(expected_error):
            catalog.delete_feature_tags(tags, feature_names)


@pytest.mark.parametrize("tags, feature_names, expected_data",
                         [
                             # [], []
                             ([], [],
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3'], dtype=str),
                                            'description': pd.Series(['d1', None, None], dtype=str),
                                            'tags': pd.Series([['t1', 't2'], ['t2'], []], dtype=object),
                                            'p1': pd.Series([None, 1., 1.], dtype=float),
                                            })),
                             # None, None
                             (None, None,
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3'], dtype=str),
                                            'description': pd.Series(['d1', None, None], dtype=str),
                                            'tags': pd.Series([[], [], []], dtype=object),
                                            'p1': pd.Series([None, 1., 1.], dtype=float),
                                            })),
                             # list tags
                             (['t1', 'xxx'], None,
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3'], dtype=str),
                                            'description': pd.Series(['d1', None, None], dtype=str),
                                            'tags': pd.Series([['t2'], ['t2'], []], dtype=object),
                                            'p1': pd.Series([None, 1., 1.], dtype=float),
                                            })),
                             # regex tags
                             ('t1', None,
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3'], dtype=str),
                                            'description': pd.Series(['d1', None, None], dtype=str),
                                            'tags': pd.Series([['t2'], ['t2'], []], dtype=object),
                                            'p1': pd.Series([None, 1., 1.], dtype=float),
                                            })),
                             ('xxx', None,
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3'], dtype=str),
                                            'description': pd.Series(['d1', None, None], dtype=str),
                                            'tags': pd.Series([['t1', 't2'], ['t2'], []], dtype=object),
                                            'p1': pd.Series([None, 1., 1.], dtype=float),
                                            })),
                             # list feature names
                             (['t2'], ['f1', 'xxx'],
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3'], dtype=str),
                                            'description': pd.Series(['d1', None, None], dtype=str),
                                            'tags': pd.Series([['t1'], ['t2'], []], dtype=object),
                                            'p1': pd.Series([None, 1., 1.], dtype=float),
                                            })),
                             # regex feature names
                             (['t2'], 'f1',
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3'], dtype=str),
                                            'description': pd.Series(['d1', None, None], dtype=str),
                                            'tags': pd.Series([['t1'], ['t2'], []], dtype=object),
                                            'p1': pd.Series([None, 1., 1.], dtype=float),
                                            })),
                             (['t2'], 'xxx',
                              pd.DataFrame({'feature_name': pd.Series(['f1', 'f2', 'f3'], dtype=str),
                                            'filename': pd.Series(['n1', 'n2', 'n3'], dtype=str),
                                            'description': pd.Series(['d1', None, None], dtype=str),
                                            'tags': pd.Series([['t1', 't2'], ['t2'], []], dtype=object),
                                            'p1': pd.Series([None, 1., 1.], dtype=float),
                                            })),
                         ])
def test_delete_feature_tags_ok_nontrivial(example_catalog4, tags, feature_names, expected_data):
    example_catalog4.delete_feature_tags(tags, feature_names)
    pd.testing.assert_frame_equal(example_catalog4.data, expected_data)
