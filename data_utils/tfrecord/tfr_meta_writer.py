import numpy as np
import os
import json

from configuration.parameter import (
    TFR_META_EXTENSION,
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
        "X_shape": X_shape,
        "X_dtype": "float32",
        "y_dtype": "int64",
        "indexes_in_datacube_dtype": "int64",
        "sample_weights_dtype": "float32",
        "samples_dtype": "int64",
        "spectrum": "int64",
        "patient_name_seperator": ",",
        "patient_index_dtype": "int64"
    }

    if len(X_shape) > 1:
        meta_data["X_patch_0"] = "int64"
        meta_data["X_patch_1"] = "int64"

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
        samples_per_patient[name]["patient_index"] = name_idx[0]

        for label, label_mask in label_masks.items():
            pat_label_mask = name_mask and label_mask
            count = np.count_nonzero(pat_label_mask)
            samples_per_patient[name][str(label)] = count

    return samples_per_patient
