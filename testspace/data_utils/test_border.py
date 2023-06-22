import pytest
import numpy as np

from data_utils.border import __get_slice, __get_axis, __get_values, __get_idx_slice, detect_border, detect_core

BOOL_ARRAY_2D = np.array([[False, False, False, False, False],
                          [False, True, True, True, False],
                          [False, True, True, True, False],
                          [False, True, True, True, False],
                          [False, False, False, False, False]])

INT_ARRAY_2D = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 2, 1, 0],
                         [0, 2, 5, 8, 0],
                         [0, 4, 5, 6, 0],
                         [0, 0, 0, 0, 0]])

FLOAT_ARRAY_2D = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.2, 3.4, 3.5, 0.0],
                           [0.0, 1.2, 3.4, 3.5, 0.0],
                           [0.0, 1.2, 3.4, 3.5, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0]])

INDEX_ARRAY_2D = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 1, 2, 3, 1, 2, 3]])

FALSE_ARRAY_2D = np.array([[False, False, False, False, False],
                           [False, False, False, False, False],
                           [False, False, False, False, False],
                           [False, False, False, False, False],
                           [False, False, False, False, False]])

SHAPE_2D = (5, 5)

RESULT_ARRAY_2D_D1_ALL_AXIS_CORE = np.array([[False, False, False, False, False],
                                             [False, False, False, False, False],
                                             [False, False, True, False, False],
                                             [False, False, False, False, False],
                                             [False, False, False, False, False]])

RESULT_ARRAY_2D_D3_ALL_AXIS_CORE = np.array([[False, False, False, False, False],
                                             [False, False, False, False, False],
                                             [False, False, False, False, False],
                                             [False, False, False, False, False],
                                             [False, False, False, False, False]])

RESULT_ARRAY_2D_D1_AXIS0_CORE = np.array([[False, False, False, False, False],
                                          [False, False, False, False, False],
                                          [False, True, True, True, False],
                                          [False, False, False, False, False],
                                          [False, False, False, False, False]])

RESULT_ARRAY_2D_D1_AXIS1_CORE = np.array([[False, False, False, False, False],
                                          [False, False, True, False, False],
                                          [False, False, True, False, False],
                                          [False, False, True, False, False],
                                          [False, False, False, False, False]])

RESULT_ARRAY_2D_D1_ALL_AXIS_BORDER = np.array([[False, False, False, False, False],
                                               [False, True, True, True, False],
                                               [False, True, False, True, False],
                                               [False, True, True, True, False],
                                               [False, False, False, False, False]])

BOOL_ARRAY_2D_TRUE = np.array([[True, True, True],
                               [True, True, True],
                               [True, True, True]])

RESULT_ARRAY_2D_TRUE_BORDER = np.array([[True, True, True],
                                        [True, False, True],
                                        [True, True, True]])

BOOL_ARRAY_1D = np.array([False, True, True, True, True, True, False])

INDEX_ARRAY_1D = np.array([1, 2, 3, 4, 5])

FALSE_ARRAY_1D = np.array([False, False, False, False, False, False, False])

SHAPE_1D = (7,)

RESULT_ARRAY_1D_D2_CORE = np.array([False, False, False, True, False, False, False])

RESULT_ARRAY_1D_D2_BORDER = np.array([False, True, True, False, True, True, False])

BOOL_ARRAY_3D = np.array([[[False, False, False], [False, False, False], [False, False, False], [False, False, False],
                           [False, False, False]],
                          [[False, False, False], [True, True, True], [True, True, True], [True, True, True],
                           [False, False, False]],
                          [[False, False, False], [True, True, True], [True, True, True], [True, True, True],
                           [False, False, False]],
                          [[False, False, False], [True, True, True], [True, True, True], [True, True, True],
                           [False, False, False]],
                          [[False, False, False], [False, False, False], [False, False, False], [False, False, False],
                           [False, False, False]]])

INDEX_ARRAY_3D = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                           [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]])

FALSE_ARRAY_3D = np.array([[[False, False, False], [False, False, False], [False, False, False], [False, False, False],
                            [False, False, False]],
                           [[False, False, False], [False, False, False], [False, False, False], [False, False, False],
                            [False, False, False]],
                           [[False, False, False], [False, False, False], [False, False, False], [False, False, False],
                            [False, False, False]],
                           [[False, False, False], [False, False, False], [False, False, False], [False, False, False],
                            [False, False, False]],
                           [[False, False, False], [False, False, False], [False, False, False], [False, False, False],
                            [False, False, False]]])

Shape_3D = (5, 5, 3)

RESULT_ARRAY_3D_D1_ALL_AXIS_CORE = np.array([[[False, False, False], [False, False, False], [False, False, False],
                                              [False, False, False], [False, False, False]],
                                             [[False, False, False], [False, False, False], [False, False, False],
                                              [False, False, False], [False, False, False]],
                                             [[False, False, False], [False, False, False], [True, True, True],
                                              [False, False, False], [False, False, False]],
                                             [[False, False, False], [False, False, False], [False, False, False],
                                              [False, False, False], [False, False, False]],
                                             [[False, False, False], [False, False, False], [False, False, False],
                                              [False, False, False], [False, False, False]]])

RESULT_ARRAY_3D_D1_ALL_AXIS_BORDER = np.array([[[False, False, False], [False, False, False], [False, False, False],
                                                [False, False, False], [False, False, False]],
                                               [[False, False, False], [True, True, True], [True, True, True],
                                                [True, True, True], [False, False, False]],
                                               [[False, False, False], [True, True, True], [True, False, True],
                                                [True, True, True], [False, False, False]],
                                               [[False, False, False], [True, True, True], [True, True, True],
                                                [True, True, True], [False, False, False]],
                                               [[False, False, False], [False, False, False], [False, False, False],
                                                [False, False, False], [False, False, False]]])

GET_VALUE_DATA = [(BOOL_ARRAY_2D, INDEX_ARRAY_2D, FALSE_ARRAY_2D, SHAPE_2D),
                  (INT_ARRAY_2D, INDEX_ARRAY_2D, FALSE_ARRAY_2D, SHAPE_2D),
                  (FLOAT_ARRAY_2D, INDEX_ARRAY_2D, FALSE_ARRAY_2D, SHAPE_2D),
                  (BOOL_ARRAY_1D, INDEX_ARRAY_1D, FALSE_ARRAY_1D, SHAPE_1D),
                  (BOOL_ARRAY_3D, INDEX_ARRAY_3D, FALSE_ARRAY_3D, Shape_3D)]


@pytest.mark.parametrize("test_array,result_values,result_new_array,result_shape", GET_VALUE_DATA)
def test__get_values(test_array, result_values, result_new_array, result_shape):
    values, new_array, shape = __get_values(array=test_array)
    assert (values == result_values).all()
    assert (new_array == result_new_array).all()
    assert shape == result_shape


GET_AXIS_DATA = [(None, (128, 128), [0, 1]),
                 ([0, 2], (128, 128, 128), [0, 2]),
                 ([], (1, 1, 1, 1), [])]


@pytest.mark.parametrize("axis,shape,result", GET_AXIS_DATA)
def test__get_axis(axis, shape, result):
    assert __get_axis(shape=shape, axis=axis) == result


GET_AXIS_ERROR_DATA = [((1, 1), [-1, 0], "For parameter 'axis' only positive integers are allowed!"),
                       ((1, 1), [0, 0.1], "For parameter 'axis' only positive integers are allowed!")]


@pytest.mark.parametrize("shape,axis,error", GET_AXIS_ERROR_DATA)
def test__get_axis_error(shape, axis, error):
    with pytest.raises(ValueError, match=error):
        __get_axis(shape=shape, axis=axis)


GET_IDX_SLICE_DATA = [(INDEX_ARRAY_2D, 0, (5, 5), 1, [0, 1], (1, 1), (slice(0, 3), slice(0, 3))),
                      (INDEX_ARRAY_2D, 3, (5, 5), 2, [0], (2, 1), (slice(0, 5), slice(1, 2)))]


@pytest.mark.parametrize("values,val_idx,shape,depth,axis,index_result,slice_result", GET_IDX_SLICE_DATA)
def test__get_idx_slice(values, val_idx, shape, axis, depth, index_result, slice_result):
    index, slices = __get_idx_slice(values=values, idx=val_idx, shape=shape, d=depth, axis=axis)
    assert index == index_result
    assert slices == slice_result


GET_SLICE_DATA = [(0, 10, 4, slice(0, 2), slice(0, 5)),
                  (5, 10, 4, slice(4, 7), slice(1, 10)),
                  (9, 10, 4, slice(8, 10), slice(5, 10))]


@pytest.mark.parametrize("index,limit,depth,result1,result2", GET_SLICE_DATA)
def test__get_slice(index, limit, depth, result1, result2):
    assert __get_slice(idx=index, limit=limit) == result1

    assert __get_slice(idx=index, limit=limit, d=depth) == result2


GET_SLICE_ERROR_DATA = [(12, 10, 1, "Index is not in array!"),
                        (5, 10, -1, "For parameter 'd' only positive integers are allowed!")]


@pytest.mark.parametrize("index,limit,depth,error", GET_SLICE_ERROR_DATA)
def test__get_slice_error_idx_higher_then_limit(index, limit, depth, error):
    with pytest.raises(ValueError, match=error):
        __get_slice(idx=index, limit=limit, d=depth)


DETECT_CORE_DATA = [(BOOL_ARRAY_1D, 2, None, RESULT_ARRAY_1D_D2_CORE),
                    (BOOL_ARRAY_2D, 1, None, RESULT_ARRAY_2D_D1_ALL_AXIS_CORE),
                    (BOOL_ARRAY_2D, 3, None, RESULT_ARRAY_2D_D3_ALL_AXIS_CORE),
                    (BOOL_ARRAY_2D, 1, [0], RESULT_ARRAY_2D_D1_AXIS0_CORE),
                    (BOOL_ARRAY_2D, 1, [1], RESULT_ARRAY_2D_D1_AXIS1_CORE),
                    (BOOL_ARRAY_3D, 1, None, RESULT_ARRAY_3D_D1_ALL_AXIS_CORE),
                    (BOOL_ARRAY_2D_TRUE, 1, None, BOOL_ARRAY_2D_TRUE)]


@pytest.mark.parametrize("in_array,depth,axis,result", DETECT_CORE_DATA)
def test_detect_core(in_array, depth, axis, result):
    if axis is None:
        assert (detect_core(in_arr=in_array, d=depth) == result).all()
    else:
        assert (detect_core(in_arr=in_array, d=depth, axis=axis) == result).all()


DETECT_BORDER_DATA = [(BOOL_ARRAY_1D, 2, None, RESULT_ARRAY_1D_D2_BORDER),
                      (BOOL_ARRAY_2D, 1, None, RESULT_ARRAY_2D_D1_ALL_AXIS_BORDER),
                      (BOOL_ARRAY_2D, 1, None, RESULT_ARRAY_2D_D1_ALL_AXIS_BORDER),
                      (BOOL_ARRAY_3D, 1, None, RESULT_ARRAY_3D_D1_ALL_AXIS_BORDER),
                      (BOOL_ARRAY_2D_TRUE, 1, None, RESULT_ARRAY_2D_TRUE_BORDER)]


@pytest.mark.parametrize("in_array,depth,axis,result", DETECT_BORDER_DATA)
def test_detect_border(in_array, depth, axis, result):
    if axis is None:
        assert (detect_border(in_arr=in_array, d=depth) == result).all()
    else:
        assert (detect_border(in_arr=in_array, d=depth, axis=axis) == result).all()
