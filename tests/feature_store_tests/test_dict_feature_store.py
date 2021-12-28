import copy

import pytest
from modelverse.feature_store.dict_feature_store import DictFeatureStore, DictFeatureTable
from modelverse.feature_store.feature_catalog import FeatureCatalog
from modelverse.feature_store.utils import FeatureTableNotFoundError
from schema import SchemaError, SchemaUnexpectedTypeError


@pytest.fixture
def ft_example0(tmp_path):
    ft = DictFeatureTable(table_path=tmp_path / '0')
    ft.create(description='example0')
    return ft


@pytest.fixture
def ft_example1(tmp_path):
    ft = DictFeatureTable(table_path=tmp_path / '1')
    ft.create(description='example1')
    ft.write_features({0: {'f1': 'nan', 'f2': 0, 'f3': 1},
                       1: {'f1': 2, 'f2': 3, 'f3': 4},
                       2: {'f1': 5, 'f2': 6, 'f3': 7},
                       })
    return ft


@pytest.fixture
def ft_example2(tmp_path):
    ft = DictFeatureTable(table_path=tmp_path / '2')
    ft.create(description='example2')
    ft.write_features({0: {'f1': 0, 'f2': 0, 'f3': 0},
                       1: {'f1': 0, 'f2': 0, 'f3': 0},
                       2: {'f1': 0, 'f2': 0, 'f3': 0},
                       })
    ft.write_features({1: {'f2': 1, 'f3': 1, 'f4': 1},
                       2: {'f2': 1, 'f3': 1, 'f4': 1},
                       3: {'f2': 1, 'f3': 1, 'f4': 1},
                       })
    return ft


@pytest.fixture
def ft_example3(tmp_path):
    ft = DictFeatureTable(table_path=tmp_path / '3')
    ft.create(description='example3')
    ft.write_features({0: {'f1': 0, 'f2': 0, 'f3': 0},
                       1: {'f1': 0, 'f2': 0, 'f3': 0},
                       2: {'f1': 0, 'f2': 0, 'f3': 0},
                       })
    ft.write_features({1: {'f2': 1, 'f3': 1, 'f4': 1},
                       2: {'f2': 1, 'f3': 1, 'f4': 1},
                       3: {'f2': 1, 'f3': 1, 'f4': 1},
                       })
    ft.write_features({1: {'f5': 2, 'f6': 2},
                       2: {'f5': 2, 'f6': 2},
                       5: {'f5': 2, 'f6': 2},
                       })
    ft.feature_catalog.write_feature_tags({'f1': ['t1'],
                                           'f6': ['t1', 't2'],
                                           })

    return ft


@pytest.mark.parametrize("ft", ['ft_example0', 'ft_example1', 'ft_example2', 'ft_example3'])
def test_ft_create(ft, request, tmp_path):
    # feature table directory exists
    (tmp_path / 'test1').mkdir(parents=True, exist_ok=False)
    tmp = DictFeatureTable(tmp_path / 'test1')
    with pytest.raises(FileExistsError):
        tmp.create()

    # feature table directory does not exist
    tmp = DictFeatureTable(tmp_path / 'test2')
    tmp.create()

    # double create
    ft = request.getfixturevalue(ft)
    with pytest.raises(FileExistsError):
        ft.create()


@pytest.mark.parametrize("ft", ['ft_example0', 'ft_example1', 'ft_example2', 'ft_example3'])
def test_ft_load(ft, request, tmp_path):
    # feature table directory does not exist
    tmp = DictFeatureTable(tmp_path / 'test1')
    with pytest.raises(FeatureTableNotFoundError):
        tmp.load()

    # feature table directory exists but has corrupt data
    (tmp_path / 'test2').mkdir(parents=True, exist_ok=False)
    tmp = DictFeatureTable(tmp_path / 'test2')
    with pytest.raises(Exception):
        tmp.load()

    # feature table directory exists and correct data
    ft = request.getfixturevalue(ft)
    m1 = copy.deepcopy(ft.table_metadata)
    c1 = copy.deepcopy(ft.feature_catalog)
    ft.load()
    m2 = copy.deepcopy(ft.table_metadata)
    c2 = copy.deepcopy(ft.feature_catalog)
    assert m1 == m2
    assert c1 == c2


@pytest.mark.parametrize("ft", ['ft_example0', 'ft_example1', 'ft_example2', 'ft_example3'])
def test_ft_delete(ft, request, tmp_path):
    ft = request.getfixturevalue(ft)
    ft.delete()
    assert len([p for p in ft.table_path.glob('*')]) == 0  # all files/dirs must be deleted
    assert ft.table_metadata == DictFeatureTable(table_path=tmp_path / 'null').table_metadata
    assert ft.feature_catalog == FeatureCatalog(ft.table_path)

    # delete -> delete, table_shape, load, read_features, write_features, etc. methods must fail
    with pytest.raises(FeatureTableNotFoundError):
        ft.delete()

    with pytest.raises(FeatureTableNotFoundError):
        ft.load()

    with pytest.raises(FeatureTableNotFoundError):
        ft.table_shape()

    with pytest.raises(FeatureTableNotFoundError):
        ft.write_features({0: {'f1': 0}})

    with pytest.raises(FeatureTableNotFoundError):
        ft.read_features()

    with pytest.raises(FeatureTableNotFoundError):
        ft.delete_features()


@pytest.mark.parametrize("ft, shape", [('ft_example0', (0, 0)), ('ft_example1', (3, 3)),
                                       ('ft_example2', (4, 4)), ('ft_example3', (5, 6))])
def test_ft_table_shape(ft, shape, request):
    ft = request.getfixturevalue(ft)
    ft.load()
    assert ft.table_shape() == shape


@pytest.mark.parametrize("data, expected_error", [([], SchemaError)])
def test_ft_write_features_fail(data, expected_error, request):
    for ft_name in ['ft_example0', 'ft_example1', 'ft_example2', 'ft_example3']:
        ft = request.getfixturevalue(ft_name)
        with pytest.raises(expected_error):
            ft.write_features(data)


@pytest.mark.parametrize("ft, data, expected_data",
                         [  # 1 denotes existing data, 2 denotes data to be inserted
                             #
                             # --------
                             # |  2   |
                             # |      |
                             # --------
                             #
                             ('ft_example0', {0: {'f1': 1, 'f2': 2}}, {0: {'f1': 1, 'f2': 2}}),
                             ('ft_example0', {'0': {'f1': '1', 'f2': 2}}, {'0': {'f1': '1', 'f2': 2}}),
                             # --------
                             # |  2   |
                             # --------
                             # --------
                             # |  2   |
                             # --------
                             ('ft_example0', {0: {'f1': 1, 'f2': 2},
                                              '0': {'f1': 1, 'f2': 2}}, {0: {'f1': 1, 'f2': 2},
                                                                         '0': {'f1': 1, 'f2': 2}}),
                             #
                             # nothing to write
                             # --------
                             # |  1   |
                             # |      |
                             # --------
                             #
                             ('ft_example1', {}, {0: {'f1': 'nan', 'f2': 0, 'f3': 1},
                                                  1: {'f1': 2, 'f2': 3, 'f3': 4},
                                                  2: {'f1': 5, 'f2': 6, 'f3': 7},
                                                  }),
                             # --------
                             # |  2 | |
                             # |____| |
                             # |      |
                             # |   1  |
                             # --------
                             #
                             ('ft_example1', {0: {'f1': 100, 'f2': 100}}, {0: {'f1': 100, 'f2': 100, 'f3': 1},
                                                                           1: {'f1': 2, 'f2': 3, 'f3': 4},
                                                                           2: {'f1': 5, 'f2': 6, 'f3': 7},
                                                                           }),
                             # --------
                             # |  1   |
                             # |   ---------
                             # |   |  |    |
                             # --------    |
                             #     |     2 |
                             #     ---------
                             #
                             ('ft_example1', {1: {'f2': -1, 'f3': -2, 'f4': -3},
                                              2: {'f2': -4, 'f3': -5, 'f4': -6},
                                              3: {'f2': -7, 'f3': -8, 'f4': -9},
                                              }, {0: {'f1': 'nan', 'f2': 0, 'f3': 1},
                                                  1: {'f1': 2, 'f2': -1, 'f3': -2, 'f4': -3},
                                                  2: {'f1': 5, 'f2': -4, 'f3': -5, 'f4': -6},
                                                  3: {'f2': -7, 'f3': -8, 'f4': -9},
                                                  }),
                             # --------
                             # | 1    |
                             # |      |
                             # |      |
                             # --------
                             #
                             #     ----------
                             #     | 2      |
                             #     ----------
                             ('ft_example1', {4: {'f2': -1, 'f3': -2, 'f4': -3},
                                              5: {'f2': -4, 'f3': -5, 'f4': -6},
                                              6: {'f2': -7, 'f3': -8, 'f4': -9},
                                              }, {0: {'f1': 'nan', 'f2': 0, 'f3': 1},
                                                  1: {'f1': 2, 'f2': 3, 'f3': 4},
                                                  2: {'f1': 5, 'f2': 6, 'f3': 7},
                                                  4: {'f2': -1, 'f3': -2, 'f4': -3},
                                                  5: {'f2': -4, 'f3': -5, 'f4': -6},
                                                  6: {'f2': -7, 'f3': -8, 'f4': -9},
                                                  }),
                             # --------
                             # |  1   |
                             # |   ---------
                             # |   |  |    |
                             # --------    |
                             #     |     2 |
                             #     ---------
                             #
                             #     ---------
                             #     |     2 |
                             #     ---------
                             #
                             ('ft_example1', {1: {'f2': -1, 'f3': -2, 'f4': -3},
                                              2: {'f2': -4, 'f3': -5, 'f4': -6},
                                              3: {'f2': -7, 'f3': -8, 'f4': -9},
                                              '5': {'f3': -8, 'f4': -9},
                                              }, {0: {'f1': 'nan', 'f2': 0, 'f3': 1},
                                                  1: {'f1': 2, 'f2': -1, 'f3': -2, 'f4': -3},
                                                  2: {'f1': 5, 'f2': -4, 'f3': -5, 'f4': -6},
                                                  3: {'f2': -7, 'f3': -8, 'f4': -9},
                                                  '5': {'f3': -8, 'f4': -9},
                                                  }),
                         ])
def test_ft_write_features_ok(ft, data, expected_data, request):
    # get ft
    ft = request.getfixturevalue(ft)
    old_metadata = copy.deepcopy(ft.table_metadata)
    old_catalog = copy.deepcopy(ft.feature_catalog)

    # write features
    ft.write_features(data)

    # read features
    ft.load()
    features, names = ft.read_features(include_untagged_features=True)

    # test features
    assert features == expected_data

    # test metadata
    assert ft.table_metadata == old_metadata

    # test catalog
    new_catalog = copy.deepcopy(ft.feature_catalog)
    new_catalog._delete_features([name for name in names if name not in old_catalog.list_feature_names()])
    assert new_catalog == old_catalog
    assert set(ft.feature_catalog.list_feature_names()) == set(old_catalog.list_feature_names() + names)


@pytest.mark.parametrize("feature_names, tags, index, include_untagged_features, expected_error",
                         [  # feature_names schema
                             ([1], None, None, False, SchemaError),
                             # tags schema
                             (None, [1], None, False, SchemaError),
                             # index schema
                             (None, None, 1, False, SchemaError),
                             # include_untagged_features schema
                             (None, None, None, 'a', SchemaUnexpectedTypeError),
                             # duplicate feature names
                             (['f1', 'f1'], None, None, False, AssertionError),
                             # index out of range
                             (None, None, [0, 7], False, AssertionError),
                         ])
def test_ft_read_features_fail(feature_names, tags, index, include_untagged_features, expected_error, request):
    for ft_name in ['ft_example0', 'ft_example1', 'ft_example2', 'ft_example3']:
        ft = request.getfixturevalue(ft_name)
        with pytest.raises(expected_error):
            ft.read_features(feature_names, tags, index, include_untagged_features)


@pytest.mark.parametrize("ft, feature_names, tags, index, include_untagged_features, expected_data",
                         [  # all features
                             ('ft_example0', None, None, None, True, {}),
                             ('ft_example3', None, None, None, True, {0: {'f1': 0, 'f2': 0, 'f3': 0},
                                                                      1: {'f1': 0, 'f2': 1, 'f3': 1, 'f4': 1, 'f5': 2,
                                                                          'f6': 2},
                                                                      2: {'f1': 0, 'f2': 1, 'f3': 1, 'f4': 1, 'f5': 2,
                                                                          'f6': 2},
                                                                      3: {'f2': 1, 'f3': 1, 'f4': 1},
                                                                      5: {'f5': 2, 'f6': 2},
                                                                      }),
                             # all features + index
                             ('ft_example3', None, None, [0, 2], True, {0: {'f1': 0, 'f2': 0, 'f3': 0},
                                                                        2: {'f1': 0, 'f2': 1, 'f3': 1, 'f4': 1, 'f5': 2,
                                                                            'f6': 2},
                                                                        }),
                             # all tagged features
                             ('ft_example3', None, None, None, False, {0: {'f1': 0},
                                                                       1: {'f1': 0, 'f6': 2},
                                                                       2: {'f1': 0, 'f6': 2},
                                                                       5: {'f6': 2},
                                                                       }),
                             # all tagged features + index
                             ('ft_example3', None, None, [0, 2], False, {0: {'f1': 0},
                                                                         2: {'f1': 0, 'f6': 2},
                                                                         }),
                             # selected feature_names
                             ('ft_example3', ['f1'], None, None, True, {0: {'f1': 0},
                                                                        1: {'f1': 0},
                                                                        2: {'f1': 0},
                                                                        }),
                             # selected feature_names + index
                             ('ft_example3', ['f1'], None, [0, 2], True, {0: {'f1': 0},
                                                                          2: {'f1': 0},
                                                                          }),
                             # selected tags with untagged excluded
                             ('ft_example3', None, ['t1'], None, False, {0: {'f1': 0},
                                                                         1: {'f1': 0, 'f6': 2},
                                                                         2: {'f1': 0, 'f6': 2},
                                                                         5: {'f6': 2},
                                                                         }),
                             # selected tags with untagged excluded + index
                             ('ft_example3', None, ['t1'], [0, 2], False, {0: {'f1': 0},
                                                                           2: {'f1': 0, 'f6': 2},
                                                                           }),
                             # selected tags with untagged included
                             ('ft_example3', None, ['t2'], None, True, {0: {'f2': 0, 'f3': 0},
                                                                        1: {'f2': 1, 'f3': 1, 'f4': 1, 'f5': 2,
                                                                            'f6': 2},
                                                                        2: {'f2': 1, 'f3': 1, 'f4': 1, 'f5': 2,
                                                                            'f6': 2},
                                                                        3: {'f2': 1, 'f3': 1, 'f4': 1},
                                                                        5: {'f5': 2, 'f6': 2},
                                                                        }),
                             # selected tags with untagged included + index
                             ('ft_example3', None, ['t2'], [0, 2], True, {0: {'f2': 0, 'f3': 0},
                                                                          2: {'f2': 1, 'f3': 1, 'f4': 1, 'f5': 2,
                                                                              'f6': 2},
                                                                          }),
                             # repeated index
                             ('ft_example3', None, ['t2'], [0, 0], True, {0: {'f2': 0, 'f3': 0}}),
                         ])
def test_ft_read_features_ok(ft, feature_names, tags, index, include_untagged_features, expected_data, request):
    ft = request.getfixturevalue(ft)
    feature_dicts, names = ft.read_features(feature_names, tags, index, include_untagged_features)
    assert feature_dicts == expected_data


@pytest.mark.skip(reason="TODO")
def test_ft_delete_features_fail():
    return


@pytest.mark.parametrize("ft, feature_names, tags, delete_untagged_features, expected_data",
                         [('ft_example3', None, None, True, {}),
                          # delete nothing
                          ('ft_example3', 'zzz', None, False, {0: {'f1': 0, 'f2': 0, 'f3': 0},
                                                               1: {'f1': 0, 'f2': 1, 'f3': 1, 'f4': 1, 'f5': 2,
                                                                   'f6': 2},
                                                               2: {'f1': 0, 'f2': 1, 'f3': 1, 'f4': 1, 'f5': 2,
                                                                   'f6': 2},
                                                               3: {'f2': 1, 'f3': 1, 'f4': 1},
                                                               5: {'f5': 2, 'f6': 2},
                                                               }),
                          # by names
                          ('ft_example3', ['f1', 'f2', 'f3', 'f4'], None, False, {0: {'f2': 0, 'f3': 0},
                                                                                  1: {'f2': 1, 'f3': 1, 'f4': 1,
                                                                                      'f5': 2, 'f6': 2},
                                                                                  2: {'f2': 1, 'f3': 1, 'f4': 1,
                                                                                      'f5': 2, 'f6': 2},
                                                                                  3: {'f2': 1, 'f3': 1, 'f4': 1},
                                                                                  5: {'f5': 2, 'f6': 2},
                                                                                  }),
                          # by names + untagged
                          ('ft_example3', ['f1', 'f2', 'f3', 'f4'], None, True, {1: {'f5': 2, 'f6': 2},
                                                                                 2: {'f5': 2, 'f6': 2},
                                                                                 5: {'f5': 2, 'f6': 2},
                                                                                 }),
                          # by tags
                          ('ft_example3', None, ['t2'], False, {0: {'f1': 0, 'f2': 0, 'f3': 0},
                                                                1: {'f1': 0, 'f2': 1, 'f3': 1, 'f4': 1, 'f5': 2},
                                                                2: {'f1': 0, 'f2': 1, 'f3': 1, 'f4': 1, 'f5': 2},
                                                                3: {'f2': 1, 'f3': 1, 'f4': 1},
                                                                5: {'f5': 2},
                                                                }),
                          # by tags + untagged
                          ('ft_example3', None, ['t2'], True, {0: {'f1': 0},
                                                               1: {'f1': 0},
                                                               2: {'f1': 0},
                                                               }),
                          ])
def test_ft_delete_features_ok(ft, feature_names, tags, delete_untagged_features, expected_data, request):
    # get ft
    ft = request.getfixturevalue(ft)
    old_metadata = copy.deepcopy(ft.table_metadata)
    old_catalog = copy.deepcopy(ft.feature_catalog)

    # delete features
    ft.delete_features(feature_names, tags, delete_untagged_features)

    # read features
    ft.load()
    new_features, new_feature_names = ft.read_features(include_untagged_features=True)

    # test features
    assert expected_data == new_features

    # test metadata
    assert ft.table_metadata == old_metadata

    # test feature catalog todo
    assert set(ft.feature_catalog.list_feature_names()).issubset(set(old_catalog.list_feature_names()))


@pytest.mark.skip(reason="TODO")
def test_ft_profile():
    return


@pytest.fixture
def fs_example0(tmp_path):
    fs = DictFeatureStore(path=tmp_path / '0')
    fs.create(description='example0')
    return fs


@pytest.fixture
def fs_example1(tmp_path):
    fs = DictFeatureStore(path=tmp_path / '1')
    fs.create(description='example1')
    fs.create_table(table_name='table0', description='table0')
    return fs


@pytest.fixture
def fs_example2(tmp_path):
    fs = DictFeatureStore(path=tmp_path / '2')
    fs.create(description='example2')
    fs.create_table(table_name='table0', description='table0')
    ft = fs.create_table(table_name='table1', description='table1')
    ft.write_features({0: {'f1': 0, 'f2': 0, 'f3': 0},
                       1: {'f1': 0, 'f2': 0, 'f3': 0},
                       2: {'f1': 0, 'f2': 0, 'f3': 0},
                       })
    ft.write_features({1: {'f2': 1, 'f3': 1, 'f4': 1},
                       2: {'f2': 1, 'f3': 1, 'f4': 1},
                       3: {'f2': 1, 'f3': 1, 'f4': 1},
                       })
    ft.write_features({1: {'f5': 2, 'f6': 2},
                       2: {'f5': 2, 'f6': 2},
                       5: {'f5': 2, 'f6': 2},
                       })
    ft.feature_catalog.write_feature_tags({'f1': ['t1'],
                                           'f6': ['t1', 't2'],
                                           })
    return fs


@pytest.mark.parametrize("fs", ['fs_example0', 'fs_example1', 'fs_example2'])
def test_fs_create(fs, request, tmp_path):
    # feature store directory exists
    (tmp_path / 'test1').mkdir(parents=True, exist_ok=False)
    tmp = DictFeatureStore(tmp_path / 'test1')
    with pytest.raises(FileExistsError):
        tmp.create()

    # feature store directory does not exist
    tmp = DictFeatureStore(path=tmp_path / 'test2')
    tmp.create()

    # double creation
    fs = request.getfixturevalue(fs)
    with pytest.raises(FileExistsError):
        fs.create()


def test_fs_load(tmp_path):
    # feature store directory does not exist
    fs = DictFeatureStore(tmp_path / 'test1')
    with pytest.raises(Exception):
        fs.load()

    # feature table directory exists but has corrupt data
    (tmp_path / 'test2').mkdir(parents=True, exist_ok=False)
    fs = DictFeatureStore(tmp_path / 'test2')
    with pytest.raises(Exception):
        fs.load()

    # feature table directory exists and correct data
    fs = DictFeatureStore(tmp_path / 'test3')
    fs.create()
    fs.load()


def test_fs_delete(tmp_path):
    fs = DictFeatureTable(tmp_path / 'test1')
    fs.create(description='example')
    fs.delete()
    assert fs.table_metadata == {'description': None,
                                 }

    with pytest.raises(Exception):
        fs.load()


@pytest.mark.skip(reason="TODO")
def test_fs_create_table():
    return


@pytest.mark.skip(reason="TODO")
def test_fs_load_table():
    return


@pytest.mark.parametrize("fs", ['fs_example1', 'fs_example2'])
def test_fs_delete_table(fs, request):
    # get feature store
    fs = request.getfixturevalue(fs)
    old_tables = fs.list_tables()

    # ok
    fs.delete_table(table_name='table0')
    assert set(fs.list_tables()) == set(old_tables) - {'table0'}
    assert set([p.name for p in fs.path.glob('*') if p.is_dir()]) == set(old_tables) - {'table0'}

    # fail
    with pytest.raises(FeatureTableNotFoundError):
        fs.delete_table(table_name='zzz')


@pytest.mark.parametrize("fs, expected_list",
                         [('fs_example0', []),
                          ('fs_example1', ['table0']),
                          ('fs_example2', ['table0', 'table1']),
                          ])
def test_fs_list_tables(fs, expected_list, request):
    fs = request.getfixturevalue(fs)
    tables = fs.list_tables()
    assert tables == expected_list

    fs.create_table(table_name='table10', description='table10')
    fs.create_table(table_name='table20', description='table20')
    fs.create_table(table_name='table30', description='table30')
    fs.load_table(table_name='table30')
    fs.delete_table(table_name='table30')
    tables = fs.list_tables()
    assert set(tables) == set(expected_list + ['table10', 'table20'])


@pytest.mark.skip(reason="TODO")
def test_summarize():
    return
