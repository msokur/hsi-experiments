import numpy as np
import inspect
from functools import wraps
from typing import Tuple

ANNOTATIONS_NOT_CHECK = [int, tuple, str, np.ndarray]


def check_params(f):
    signature = inspect.signature(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        bound_args = signature.bind(*args, **kwargs)
        for name, value in bound_args.arguments.items():
            annotation = signature.parameters[name].annotation
            if annotation is inspect.Signature.empty:
                continue
            elif annotation in ANNOTATIONS_NOT_CHECK:
                if not isinstance(value, annotation):
                    raise TypeError(f"Parameter '{name}' is not from the type {annotation}")
                continue
            annotation(name, value)
        return f(*args, **kwargs)
    return wrapper


def __list_positive(name, value):
    if value is not None:
        if not isinstance(value, list):
            raise TypeError(f"For parameter {name} only List are allowed")
        if not (all(isinstance(v, int) and __positive(name, v) for v in value)):
            __value_error(name, "positive integers")


def __positive(name, value):
    if value < 0:
        __value_error(name, "positive integers")
    return True


def __value_error(param: str, typ: str):
    raise ValueError(f"For parameter '{param}' only {typ} are allowed!")


def print_array(array: np.ndarray):
    shape = array.shape
    for sub in array:
        if shape.__len__() > 1:
            print_array(sub)
            print()
        else:
            print(sub, end="\t")


@check_params
def detect_core(in_arr: np.ndarray, d: __positive = 1, axis: __list_positive = None) -> np.ndarray:
    """ Detects the core in a array.

    detect_core(in_arr: np.ndarray, d: int, axis: List[int])

    If no depth given, the border around the core will remove with the depth 1.

    :param in_arr: The input array  where the core should detect.
    :param d: Is the depth from the border that should remove around the core.
        There are only positive integers are allowed.
    :param axis: A list with positive integer. Only the axis in that list will recognize. Axis 0 is the direction along
        rows, axis 1 is the direction along columns and so on.

    :return: Returns a boolean array withe the core.

    Example
    -------
    >>> a = np.full((5, 5), False)
    >>> a[1:5][1:5] = True
    array( [[False, False,  False,  False,  False],
    ...     [False, True,   True,   True,   False],
    ...     [False, True,   True,   True,   False],
    ...     [False, True,   True,   True,   False],
    ...     [False, False,  False,  False,  False]])
    >>> detect_core(a)
    array( [[False, False,  False,  False,  False],
    ...     [False, False,  False,  False,  False],
    ...     [False, False,  True,   False,  False],
    ...     [False, False,  False,  False,  False],
    ...     [False, False,  False,  False,  False]])
    >>> detect_core(a, axis=[0])
    array( [[False, False,  False,  False,  False],
    ...     [False, False,  False,  False,  False],
    ...     [False, True,   True,   True,   False],
    ...     [False, False,  False,  False,  False],
    ...     [False, False,  False,  False,  False]])
    >>> detect_core(a, axis=[1])
    array( [[False, False,  False,  False,  False],
    ...     [False, False,  True,   False,  False],
    ...     [False, False,  True,   False,  False],
    ...     [False, False,  True,   False,  False],
    ...     [False, False,  False,  False,  False]])
    >>> detect_core(a, d=2)
    array( [[False, False,  False,  False,  False],
    ...     [False, False,  False,  False,  False],
    ...     [False, False,  False,  False,  False],
    ...     [False, False,  False,  False,  False],
    ...     [False, False,  False,  False,  False]])
    """
    values, new_array, shape = __get_values(in_arr)
    axis = __get_axis(shape=shape, axis=axis)
    for idx in np.arange(len(values[0])):
        index, slices = __get_idx_slice(values=values, idx=idx, shape=shape, d=d, axis=axis)
        new_array[index] = np.all(in_arr[slices])

    return new_array


@check_params
def detect_border(in_arr: np.ndarray, d: __positive = 1, axis: __list_positive = None) -> np.ndarray:
    """ Detects the border in a array.

        detect_border(in_arr: np.ndarray, d: int, axis: List[int]).

        If no given, the depth has a default value of 1.

        :param in_arr: The input array  where the border should detect.
        :param d: Is the depth from the border. There are only positive integers are allowed.
        :param axis: A list with positive integer. Only the axis in that list will recognize. Axis 0 is the direction
            along rows, axis 1 is the direction along columns and so on.

        :return: Returns a boolean array withe the border.

        Example
        -------
        >>> a = np.full((5, 5), False)
        >>> a[1:5][1:5] = True
        array( [[False, False,  False,  False,  False],
        ...     [False, True,   True,   True,   False],
        ...     [False, True,   True,   True,   False],
        ...     [False, True,   True,   True,   False],
        ...     [False, False,  False,  False,  False]])
        >>> detect_border(a)
        array( [[False, False,  False,  False,  False],
        ...     [False, True,   True,   True,   False],
        ...     [False, True,   False,  True,   False],
        ...     [False, True,   True,   True,   False],
        ...     [False, False,  False,  False,  False]])
        >>> detect_border(a, axis=[0])
        array( [[False, False,  False,  False,  False],
        ...     [False, True,   True,   True,   False],
        ...     [False, False,  False,  False,  False],
        ...     [False, True,   True,   True,   False],
        ...     [False, False,  False,  False,  False]])
        >>> detect_border(a, axis=[1])
        array( [[False, False,  False,  False,  False],
        ...     [False, True,   False,  True,   False],
        ...     [False, True,   False,  True,   False],
        ...     [False, True,   False,  True,   False],
        ...     [False, False,  False,  False,  False]])
        >>> detect_border(a, d=2)
        array( [[False, False,  False,  False,  False],
        ...     [False, True,   True,   True,   False],
        ...     [False, True,   True,   True,   False],
        ...     [False, True,   True,   True,   False],
        ...     [False, False,  False,  False,  False]])
        """
    val, new_array, shape = __get_values(in_arr)
    axis = __get_axis(shape=shape, axis=axis)
    for idx in np.arange(len(val[0])):
        index, slices = __get_idx_slice(values=val, idx=idx, shape=shape, d=d, axis=axis)
        if any([index[i] == 0 or index[i] == shape[i] - 1 for i in range(len(index))]):
            new_array[index] = True
        else:
            new_array[index] = np.all(in_arr[slices]) ^ np.any(in_arr[slices])

    return new_array


def __get_values(array: np.ndarray):
    shape = array.shape
    values = np.where(array)
    new_array = np.full(shape, False)

    return values, new_array, shape


@check_params
def __get_axis(shape: tuple, axis: __list_positive = None) -> list:
    if axis is not None:
        axis = list(set(axis))
    else:
        axis = list(range(shape.__len__()))

    return axis


def __get_idx_slice(values: np.ndarray, idx: int, shape: tuple, d: int, axis: list) -> Tuple[tuple, tuple]:
    slices = ()
    index = ()
    for ax in np.arange(len(values)):
        if ax not in axis:
            slices += ((__get_slice(idx=int(values[ax][idx]), limit=shape[ax], d=0)),)
        else:
            slices += ((__get_slice(idx=int(values[ax][idx]), limit=shape[ax], d=d)),)
        index += (values[ax][idx],)

    return index, slices


@check_params
def __get_slice(idx: int, limit: int, d: __positive = 1) -> slice:
    if idx >= limit:
        raise ValueError("Index is not in array!")
    i_start, i_end = idx - d, idx + d + 1
    if i_start < 0:
        i_start = 0
    if i_end > limit:
        i_end = limit

    return slice(i_start, i_end)
