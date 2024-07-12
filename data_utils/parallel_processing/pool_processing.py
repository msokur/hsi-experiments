import os
import multiprocessing as mp

from functools import partial


def start_pool_processing(map_func, parallel_args: list, is_on_cluster: bool, fix_args: list = None,
                          print_out: str = None):
    """ Create a pool parallel processing pipeline

    :param map_func: The function with the parallel pipeline.
    :param parallel_args: Arguments that will be split to the different processes.
    :param is_on_cluster: True if the parallel processing is executing on cluster.
    :param fix_args: Arguments that are the same for every process.
    :param print_out: E.g. the function name to print.

    :return: A list with the length of the number of parallel processes and the result for every process
        (if the map_func returns some values).


    Note
    ------
    The fix arguments must be in the first argument that passed to the function.

    """
    if is_on_cluster:
        cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    else:
        cpus = mp.cpu_count()

    parallel_args_length = len(parallel_args[0])
    if parallel_args_length < cpus:
        cpus = parallel_args_length

    if len(parallel_args) > 1:
        for args in parallel_args[1:]:
            assert len(args) == parallel_args_length, "Check your parallel args, they have not the same length!"

    parallel_args = [args for args in zip(*parallel_args)]

    if print_out is not None:
        print(f"----{print_out} parallel processing with {cpus} processes!----")

    if fix_args is not None:
        map_func = partial(map_func,
                           *fix_args)

    with mp.Pool(processes=cpus) as pool:
        result = pool.starmap(map_func,
                              parallel_args)

    return result
