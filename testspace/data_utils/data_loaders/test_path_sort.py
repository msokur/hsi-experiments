import pytest

from data_utils.data_loaders.path_sort import get_sort, get_number, sort

DATE_PATHS = ["C:\\folder1\\folder2\\2023_07_04_10_10_30_cube.mat",
              "C:\\folder1\\folder2\\2022_01_30_12_30_01_cube.mat",
              "C:\\folder1\\folder2\\2023_07_04_11_30_30_cube.mat",
              "C:\\folder1\\folder2\\2022_10_10_01_30_50_cube.mat"]

RESULT_DATE_PATHS = ["C:\\folder1\\folder2\\2022_01_30_12_30_01_cube.mat",
                     "C:\\folder1\\folder2\\2022_10_10_01_30_50_cube.mat",
                     "C:\\folder1\\folder2\\2023_07_04_10_10_30_cube.mat",
                     "C:\\folder1\\folder2\\2023_07_04_11_30_30_cube.mat"]

NUMBER_PATHS_1 = ["C:\\folder1\\folder2\\data4cube.dat",
                  "C:\\folder1\\folder2\\data2cube.dat",
                  "C:\\folder1\\folder2\\data1cube.dat",
                  "C:\\folder1\\folder2\\data3cube.dat"]

NUMBER_SPLITS_1 = ["data", "cube"]

RESULT_NUMBER_PATH_1 = ["C:\\folder1\\folder2\\data1cube.dat",
                        "C:\\folder1\\folder2\\data2cube.dat",
                        "C:\\folder1\\folder2\\data3cube.dat",
                        "C:\\folder1\\folder2\\data4cube.dat"]

NUMBER_PATHS_2 = ["C:\\folder1\\folder2\\data4.dat",
                  "C:\\folder1\\folder2\\data2.dat",
                  "C:\\folder1\\folder2\\data1.dat",
                  "C:\\folder1\\folder2\\data3.dat"]

NUMBER_SPLITS_2 = ["data", None]

RESULT_NUMBER_PATH_2 = ["C:\\folder1\\folder2\\data1.dat",
                        "C:\\folder1\\folder2\\data2.dat",
                        "C:\\folder1\\folder2\\data3.dat",
                        "C:\\folder1\\folder2\\data4.dat"]

NUMBER_PATHS_3 = ["C:\\folder1\\folder2\\4cube.dat",
                  "C:\\folder1\\folder2\\2cube.dat",
                  "C:\\folder1\\folder2\\1cube.dat",
                  "C:\\folder1\\folder2\\3cube.dat"]

NUMBER_SPLITS_3 = [None, "cube"]

RESULT_NUMBER_PATH_3 = ["C:\\folder1\\folder2\\1cube.dat",
                        "C:\\folder1\\folder2\\2cube.dat",
                        "C:\\folder1\\folder2\\3cube.dat",
                        "C:\\folder1\\folder2\\4cube.dat"]

GET_SORT_DATA = [(DATE_PATHS, False, None, RESULT_DATE_PATHS),
                 (NUMBER_PATHS_1, True, NUMBER_SPLITS_1, RESULT_NUMBER_PATH_1),
                 ([], False, None, []),
                 ([], True, None, [])]


@pytest.mark.parametrize("paths,number_split,splits,result", GET_SORT_DATA)
def test_get_sort(paths, number_split, splits, result):
    assert get_sort(paths=paths, number=number_split, split=splits) == result


GET_NUMBER_DATA = [(NUMBER_PATHS_1[0], NUMBER_SPLITS_1, 4),
                   ("C:\\folder1\\folder2\\test.zarr/1cube", [None, "cube"], 1)]


@pytest.mark.parametrize("elem,split,result", GET_NUMBER_DATA)
def test_get_number(elem, split, result):
    assert get_number(elem=elem, split=split) == result


GET_NUMBER_ERROR_DATA = [(NUMBER_PATHS_1[0], ["data", "ube"], ValueError,
                          "Can't change literal to integer. Check your 'splits' parameter!"),
                         (NUMBER_PATHS_1[1], None, TypeError, "'NoneType' object is not subscriptable"),
                         ("", NUMBER_SPLITS_1, ValueError,
                          "Can't change literal to integer. Check your 'splits' parameter!")]


@pytest.mark.parametrize("elem,splits,error,message", GET_NUMBER_ERROR_DATA)
def test_get_number_error(elem, splits, error, message):
    with pytest.raises(error, match=message):
        get_number(elem=elem, split=splits)


SORT_DATA = [(NUMBER_PATHS_1, NUMBER_SPLITS_1, RESULT_NUMBER_PATH_1),
             (NUMBER_PATHS_2, NUMBER_SPLITS_2, RESULT_NUMBER_PATH_2),
             (NUMBER_PATHS_3, NUMBER_SPLITS_3, RESULT_NUMBER_PATH_3)]


@pytest.mark.parametrize("paths,number_splits,result", SORT_DATA)
def test_sort(paths, number_splits, result):
    assert sort(paths=paths, split=number_splits) == result
