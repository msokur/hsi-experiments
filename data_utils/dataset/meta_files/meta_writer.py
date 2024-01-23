import numpy as np
import os
import json

from configuration.parameter import (
    TFR_TYP, GEN_TYP, TFR_META_EXTENSION, GEN_META_EXTENSION, SAMPLES_PER_NAME, TOTAL_SAMPLES, FEATURE_X, FEATURE_Y,
    FEATURE_IDX_CUBE, FEATURE_WEIGHTS, FEATURE_SAMPLES, FEATURE_SPEC, FEATURE_PAT_NAME, PAT_NAME_SEPERATOR,
    PAT_NAME_ENCODING, FEATURE_PAT_IDX, FEATURE_X_AXIS_1, FEATURE_X_AXIS_2
)


def write_meta_info(save_dir: str, file_name: str, labels: np.ndarray, names: np.ndarray, names_idx: np.ndarray,
                    X_shape: tuple, typ: str):
    """Create a meta file for shuffled data.

    The meta file contains the data typ from the different dates in the shuffled file, the shape from X and the count
    from every name per label.

    You can choose between meta files for TFRecord files or for shuffled files for data generator.
    For TFRecord meta file use the key 'tfr' and for the data generator 'generator'.

    :param save_dir: Path to save the meta file
    :param file_name: Name from the meta file
    :param labels: Array with labels to count
    :param names: Array with names
    :param names_idx: Array with the indexes for the names
    :param X_shape: The shape from the X data
    :param typ: Type of meta file

    :raise ValueError: If the typ for the meta file wrong or too many indexes for one name
    """
    extension = _check_meta_typ(typ=typ)

    meta_data = _base_meta_data(total_samples=X_shape[0], X_shape=list(X_shape)[1:], typ=typ)

    meta_data[SAMPLES_PER_NAME] = _count_labels_per_name(labels=labels, names=names, names_idx=names_idx)

    with open(os.path.join(save_dir, file_name + extension), "w") as file:
        json.dump(meta_data, file)


def _check_meta_typ(typ: str) -> str:
    if typ == TFR_TYP:
        return TFR_META_EXTENSION
    elif typ == GEN_TYP:
        return GEN_META_EXTENSION
    else:
        raise ValueError("Wrong type for meta file, check your configurations!")


def _base_meta_data(total_samples: int, X_shape: list, typ: str) -> dict:
    meta_data = {
        TOTAL_SAMPLES: total_samples,
        f"{FEATURE_X}_shape": X_shape,
        f"{FEATURE_X}_dtype": "float32",
        f"{FEATURE_Y}_dtype": "int64",
        f"{FEATURE_IDX_CUBE}_dtype": "int64",
        f"{FEATURE_WEIGHTS}_dtype": "float32",
        f"{FEATURE_PAT_IDX}_dtype": "int64"
    }

    if typ == "tfr":
        meta_data[f"{FEATURE_SAMPLES}_dtype"] = "int64"
        meta_data[f"{FEATURE_SPEC}_size_dtype"] = "int64"
        meta_data[f"{FEATURE_PAT_NAME}_seperator"] = PAT_NAME_SEPERATOR
        meta_data[f"{FEATURE_PAT_NAME}_encoding"] = PAT_NAME_ENCODING

    if len(X_shape) > 1:
        meta_data[f"{FEATURE_X_AXIS_1}_size_dtype"] = "int64"
        meta_data[f"{FEATURE_X_AXIS_2}_size_dtype"] = "int64"

    return meta_data


def _count_labels_per_name(labels: np.ndarray, names: np.ndarray, names_idx: np.ndarray) -> dict:
    samples_per_patient = {}
    unique_names = np.unique(names)
    unique_labels = np.unique(labels)
    label_masks = {label: np.isin(labels, label) for label in unique_labels}

    for name in unique_names:
        samples_per_patient[name] = {}
        name_mask = np.isin(names, name)
        name_idx = np.unique(names_idx[name_mask])
        if len(name_idx) > 1:
            raise ValueError(f"Too many indexes for patient name '{name}'!")
        samples_per_patient[name][FEATURE_PAT_IDX] = int(name_idx[0])

        for label, label_mask in label_masks.items():
            pat_label_mask = np.logical_and(label_mask, name_mask)
            count = np.count_nonzero(pat_label_mask)
            if count > 0:
                samples_per_patient[name][str(label)] = count

    return samples_per_patient
