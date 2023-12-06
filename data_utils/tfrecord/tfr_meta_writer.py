import numpy as np
import os
import json

from configuration.parameter import (
    TFR_META_EXTENSION, FEATURE_X, FEATURE_Y, FEATURE_IDX_CUBE, FEATURE_WEIGHTS, FEATURE_SAMPLES, FEATURE_SPEC,
    FEATURE_PAT_NAME, PAT_NAME_SEPERATOR, PAT_NAME_ENCODING, FEATURE_PAT_IDX, FEATURE_X_AXIS_1, FEATURE_X_AXIS_2
)


def write_meta_info(shuffle_saving_path: str, file_name: str, labels: list, names: list, names_idx: list,
                    X_shape: tuple):
    X_shape = list(X_shape)[1:]

    meta_data = _base_meta_data(X_shape=X_shape)

    meta_data["samples_per_patient_name"] = _count_labels_per_name(labels=labels, names=names, names_idx=names_idx)

    with open(os.path.join(shuffle_saving_path, file_name + TFR_META_EXTENSION), "w") as file:
        json.dump(meta_data, file)


def _base_meta_data(X_shape: list) -> dict:
    meta_data = {
        f"{FEATURE_X}_shape": X_shape,
        f"{FEATURE_X}_dtype": "float32",
        f"{FEATURE_Y}_dtype": "int64",
        f"{FEATURE_IDX_CUBE}_dtype": "int64",
        f"{FEATURE_WEIGHTS}_dtype": "float32",
        f"{FEATURE_SAMPLES}_dtype": "int64",
        f"{FEATURE_SPEC}_size_dtype": "int64",
        f"{FEATURE_PAT_NAME}_seperator": PAT_NAME_SEPERATOR,
        f"{FEATURE_PAT_NAME}_encoding": PAT_NAME_ENCODING,
        f"{FEATURE_PAT_IDX}_dtype": "int64"
    }

    if len(X_shape) > 1:
        meta_data[f"{FEATURE_X_AXIS_1}_size_dtype"] = "int64"
        meta_data[f"{FEATURE_X_AXIS_2}_size_dtype"] = "int64"

    return meta_data


def _count_labels_per_name(labels: list, names: list, names_idx: list) -> dict:
    samples_per_patient = {}
    unique_names = np.unique(names)
    unique_labels = np.unique(labels)
    label_masks = {label: np.isin(labels, label) for label in unique_labels}

    for name in unique_names:
        name_mask = np.isin(names, name)
        name_idx = np.unique(names_idx[name_mask])
        if len(name_idx) > 1:
            raise ValueError(f"Too many indexes for patient name '{name}'!")
        samples_per_patient[name][FEATURE_PAT_IDX] = name_idx[0]

        for label, label_mask in label_masks.items():
            pat_label_mask = name_mask and label_mask
            count = np.count_nonzero(pat_label_mask)
            samples_per_patient[name][str(label)] = count

    return samples_per_patient
