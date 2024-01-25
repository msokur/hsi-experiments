import json
import os
from typing import Dict, Tuple, List
import numpy as np

from configuration.parameter import (
    FEATURE_X, SAMPLES_PER_NAME, FEATURE_PAT_IDX, DATASET_TYPE, TFR_TYP, GEN_TYP, TFR_META_EXTENSION, GEN_META_EXTENSION
)


def get_cw_from_meta(files: list, labels: list, names: list, dataset_typ: str = DATASET_TYPE) -> Dict[str, float]:
    """Calculate class weights from meta file.

    :param files: Paths to datasets
    :param labels: Used labels
    :param names: Used names
    :param dataset_typ: Typ of the dataset

    :return: A dictionary with the label name as str and the class weight
    """
    meta_files = get_meta_files(paths=files, typ=dataset_typ)

    sums = {f"{label}": 0.0 for label in labels}

    for samples_per_labels in _get_samples_per_label(paths=meta_files, names=names):
        for label, v in samples_per_labels.items():
            if label in sums.keys():
                sums[label] += v

    total = np.sum([samples for samples in sums.values()])
    cw = {}
    for label, samples in sums.items():
        if samples > 0.0:
            cw[label] = (1 / samples) * total / labels.__len__()
        else:
            cw[label] = 0.0

    return cw


def get_shape_from_meta(files: list, dataset_type: str = DATASET_TYPE) -> Tuple[int]:
    """Get Dataset shape for X from meta files

    :param files: Paths to datasets
    :param dataset_type: Typ of the dataset

    :return: A tuple with the shape for X

    :raises ValueError: When different shape in the meta files
    """
    meta_files = get_meta_files(paths=files, typ=dataset_type)

    model_shape = None
    for p in meta_files:
        if model_shape is None:
            model_shape = get_section_from_meta(file_path=p, section=f"{FEATURE_X}_shape")
        else:
            if model_shape != get_section_from_meta(file_path=p, section=f"{FEATURE_X}_shape"):
                raise ValueError("Chck your dataset! Different data shapes in meta files!")

    return tuple(model_shape)


def get_meta_files(paths: List[str], typ: str) -> List[str]:
    """Replace TFRecord file extension with TFRecord meta file extension.

    :param paths: File paths
    :param typ: Type of the meta files

    :return: List with paths to TFRecord meta files

    :raise ValueError: When the typ for the meta file is wrong
    """
    if typ == GEN_TYP:
        extension = GEN_META_EXTENSION
    elif typ == TFR_TYP:
        extension = TFR_META_EXTENSION
    else:
        raise ValueError("Wrong meta file typ to load!")
    return [os.path.splitext(p)[0] + extension for p in paths]


def _get_samples_per_label(paths: List[str], names: list):
    for p in paths:
        samples_per_names = get_section_from_meta(file_path=p, section=SAMPLES_PER_NAME)
        # create a list with samples per label for the needed names
        samples_per_labels = [samples_per_names[name] for name in names if name in names]
        for samples_per_label in samples_per_labels:
            # remove patient index
            samples_per_label.pop(FEATURE_PAT_IDX)
            yield samples_per_label


def get_section_from_meta(file_path: str, section: str) -> dict:
    # open meta file
    info = json.load(open(file=file_path, mode="r"))
    # get the section with the samples per name
    return info[section]
