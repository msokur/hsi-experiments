from typing import Dict

import numpy as np

from .meta_files import get_meta_files, get_section_from_meta

from configuration.parameter import (
    SAMPLES_PER_NAME, FEATURE_PAT_IDX
)


def parse_names_to_int(files: list, meta_type: str) -> Dict[str, int]:
    """Read the integer values in the metafile for every name.

    :param files: Paths to datasets
    :param meta_type: Type of the meta files

    :return: A dictionary with the actual str name as key and the integer as value

    :raises ValueError: If there is more than one integer for a name
    """
    meta_files = get_meta_files(paths=files, typ=meta_type)

    names_int = {}
    for meta_file in meta_files:
        samples_per_names = get_section_from_meta(file_path=meta_file, section=SAMPLES_PER_NAME)
        for name in samples_per_names.keys():
            # get indexes from all meta files
            if name not in names_int:
                names_int[name] = [samples_per_names[name][FEATURE_PAT_IDX]]
            else:
                names_int[name] += [samples_per_names[name][FEATURE_PAT_IDX]]

    for name, indexes in names_int.items():
        unique_idx = np.unique(indexes)
        if unique_idx.shape[0] == 1:
            names_int[name] = int(unique_idx[0])
        else:
            raise ValueError(f"Too many patient indexes in meta files for the name '{name}'!")

    return names_int
