import pytest

from data_utils.parallel_processing.pool_processing import start_pool_processing

PARALLEL_ARG_ONE = [1, 2, 3, 4, 5, 6, 7, 8]
PARALLEL_ARG_TWO = [2, 4, 6, 8, 10, 12, 14, 16]
FIX_ARG_ONE = 2
FIX_ARG_TWO = 4


def func_one(fix_arg: int, parallel_arg: int) -> int:
    return parallel_arg * fix_arg


def func_two(fix_arg: int, parallel_arg_one: int, parallel_arg_two: int) -> float:
    return parallel_arg_one / parallel_arg_two * fix_arg


def func_three(fix_arg_one: int, fix_arg_two: int, parallel_arg: int) -> float:
    return parallel_arg * fix_arg_one / fix_arg_two


def func_four(fix_arg_one: int, fix_arg_two: int, parallel_arg_one: int, parallel_arg_two: int) -> float:
    return parallel_arg_one / parallel_arg_two * fix_arg_one / fix_arg_two


class SomeClass:
    def __init__(self, fix_number: int):
        self.fix_number = fix_number

    def some_fuc(self, number):
        return number ** 2 * self.fix_number


POOL_DATA = [(func_one, [PARALLEL_ARG_ONE], [FIX_ARG_ONE], [2, 4, 6, 8, 10, 12, 14, 16]),
             (func_two, [PARALLEL_ARG_ONE, PARALLEL_ARG_TWO], [FIX_ARG_ONE], [1., 1., 1., 1., 1., 1., 1., 1.]),
             (func_three, [PARALLEL_ARG_ONE], [FIX_ARG_ONE, FIX_ARG_TWO], [0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.]),
             (func_four, [PARALLEL_ARG_ONE, PARALLEL_ARG_TWO], [FIX_ARG_ONE, FIX_ARG_TWO],
              [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]),
             (SomeClass(fix_number=2).some_fuc, [[1, 2, 3]], None, [2, 8, 18])]


@pytest.mark.parametrize("map_func,parallel_args,fix_args,results", POOL_DATA)
def test_start_pool_processing(test_config, map_func, parallel_args, fix_args, results):
    pool_results = start_pool_processing(map_func=map_func,
                                         parallel_args=parallel_args,
                                         fix_args=fix_args,
                                         is_on_cluster=test_config.CLUSTER)

    assert pool_results == results


def test_start_pool_processing_arg_length_error():
    with pytest.raises(AssertionError, match="Check your parallel args, they have not the same length!"):
        start_pool_processing(map_func=func_one,
                              parallel_args=[[0, 1, 2], [0, 1]],
                              fix_args=[1],
                              is_on_cluster=True)
