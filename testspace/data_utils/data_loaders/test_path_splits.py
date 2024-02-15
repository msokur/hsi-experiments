import pytest
import numpy as np
import os

from data_utils.data_loaders.path_splits import get_splits, split_int, split_list, split_name


PATHS = [os.path.join(*["C:", "folder", "data0_1.dat"]), os.path.join(*["C:", "folder", "data1_1.dat"]),
         os.path.join(*["C:", "folder", "data0_2.dat"]), os.path.join(*["C:", "folder", "data1_2.dat"]),
         os.path.join(*["C:", "folder", "data10_1.dat"]), os.path.join(*["C:", "folder", "data10_2.dat"])]

PATHS2 = [os.path.join(*["C:", "folder", "data0.dat"]), os.path.join(*["C:", "folder", "data1.dat"]),
          os.path.join(*["C:", "folder", "data2.dat"]), os.path.join(*["C:", "folder", "data10.dat"]),
          os.path.join(*["C:", "folder", "data20.dat"]), os.path.join(*["C:", "folder", "data22.dat"])]


GET_SPLITS_DATA = [("Name", PATHS, (0, 5), [np.array([0, 2]), np.array([1, 3, 4, 5])]),
                   ("List", PATHS, [["data0_1", "data0_2"], ["data1_1", "data1_2"]],
                    [np.array([0, 2]), np.array([1, 3])]),
                   ("Files", PATHS, 2, [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])])]


@pytest.mark.parametrize("typ,paths,values,result", GET_SPLITS_DATA)
def test_get_splits(typ, paths, values, result):
    value = get_splits(typ=typ, paths=paths, values=values)

    assert len(value) == len(result)

    for val, res in zip(value, result):
        assert (val == res).all()


GET_SPLITS_WARNING_DATA = [("Nothing", 2,
                            "'SPLIT_PATHS_BY' in configurations is not correct. Paths will be split by files with 1!"),
                           ("Name", 2,
                            "'CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST' in configurations is not correct. Paths will be "
                            "split by files with 1!"),
                           ("List", 2,
                            "'CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST' in configurations is not correct. Paths will be "
                            "split by files with 1!"),
                           ("Files", (0, 5),
                            "'CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST' in configurations is not correct. Paths will be "
                            "split by files with 1!")]


@pytest.mark.parametrize("typ,values,message", GET_SPLITS_WARNING_DATA)
def test_get_splits_warning(typ, values, message):
    with pytest.warns(UserWarning, match=message):
        value = get_splits(typ=typ, paths=PATHS, values=values)

    result = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array(4), np.array(5)]

    assert len(value) == len(result)

    for val, res in zip(value, result):
        assert (val == res).all()


SPLIT_INT_DATA = [(10, 2, [np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), np.array([6, 7]), np.array([8, 9])]),
                  (10, 3, [np.array([0, 1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]),
                  (11, 3, [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10])])]


@pytest.mark.parametrize("length,number,result", SPLIT_INT_DATA)
def test_split_int(length, number, result):
    value = split_int(path_length=length, num=number)

    assert len(value) == len(result)

    for val, res in zip(value, result):
        assert (val == res).all()


def test_split_int_error():
    with pytest.raises(ValueError):
        split_int(path_length=3, num=4)


SPLIT_LIST_DATA = [(PATHS, [["data0_1", "data0_2"], ["data1_1", "data1_2"]], [np.array([0, 2]), np.array([1, 3])]),
                   (PATHS, [["data0_1"], ["data0_2"], ["data1_1"], ["data1_2"]],
                    [np.array([0]), np.array([2]), np.array([1]), np.array([3])]),
                   (PATHS2, [["data0"], ["data1"], ["data2"]], [np.array([0]), np.array([1]), np.array([2])])]


@pytest.mark.parametrize("paths,names_list,result", SPLIT_LIST_DATA)
def test_split_list(paths, names_list, result):
    values = split_list(paths=paths, name_list=names_list)
    assert len(values) == len(result)

    for val, res in zip(values, result):
        assert (val == res).all()


SPLIT_LIST_WARNING_DATA = [([["test"], ["data0_1"], ["data0_2"], ["data1_1"], ["data1_2"]],
                            [np.array([0]), np.array([2]), np.array([1]), np.array([3])]),
                           ([["data0_1", "test", "data0_2"], ["data1_1", "data1_2"]],
                           [np.array([0, 2]), np.array([1, 3])])]


@pytest.mark.parametrize("name_list,result", SPLIT_LIST_WARNING_DATA)
def test_split_list_warning(name_list, result):
    with pytest.warns(UserWarning, match="No matches with name 'test' in paths. The name will be ignored!"):
        values = split_list(paths=PATHS, name_list=name_list)

    for val, res in zip(values, result):
        assert (val == res).all()


def test_split_list_error():
    with pytest.raises(ValueError, match="More than one match with the name 'data0_1'. Check your split name list!"):
        paths = PATHS + [os.path.join(*["C:", "folder", "data0_1.dat"])]
        split_list(paths=paths, name_list=[["data0_1"], ["data1_1", "data1_2"]])


SPLIT_NAME_DATA = [(PATHS, (0, 5), [np.array([0, 2]), np.array([1, 3, 4, 5])]),
                   (PATHS, (0, 15), [np.array([0]), np.array([2]), np.array([4]), np.array([5]), np.array([1]),
                                     np.array([3])]),
                   ([], (0, 5), [])]


@pytest.mark.parametrize("paths,name_slice,result", SPLIT_NAME_DATA)
def test_split_name(paths, name_slice, result):
    values = split_name(paths=paths, name_slice=name_slice)

    assert len(values) == len(result)

    for val, res in zip(values, result):
        assert (val == res).all()
