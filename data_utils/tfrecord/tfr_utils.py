import json

from typing import List, Dict

import numpy as np
import tensorflow as tf

from configuration.parameter import (
    TFR_META_EXTENSION, TFR_FILE_EXTENSION, SAMPLES_PER_NAME, FEATURE_PAT_IDX,
)


def get_cw_from_meta(tfr_files: list, labels: list, names: list) -> Dict[int, float]:
    meta_files = _get_meta_files(tfr_paths=tfr_files)

    sums = {f"{label}": 0.0 for label in labels}

    for samples_per_labels in _get_samples_per_label(paths=meta_files, names=names):
        for label, v in samples_per_labels.items():
            if label in sums.keys():
                sums[label] += v

    total = np.sum([samples for samples in sums.values()])
    cw = {}
    for label, samples in sums.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            cw[label] = (1 / samples) * total / labels.__len__()
        if cw[label] == np.inf:
            cw[label] = 0.0

    return cw


def parse_names_to_int(tfr_files: list) -> Dict[str, int]:
    meta_files = _get_meta_files(tfr_paths=tfr_files)

    names_int = {}
    for meta_file in meta_files:
        samples_per_names = _get_samples_per_name(file_path=meta_file)
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


def select(data, allowed):
    use = tf.map_fn(fn=lambda com: tf.equal(x=data, y=allowed), elems=allowed, fn_output_signature=tf.bool)
    return tf.math.reduce_any(input_tensor=use)


def _get_meta_files(tfr_paths: List[str]) -> List[str]:
    return [tfr_p.replace(TFR_FILE_EXTENSION, TFR_META_EXTENSION) for tfr_p in tfr_paths]


def _get_samples_per_label(paths: List[str], names: list):
    for p in paths:
        samples_per_names = _get_samples_per_name(file_path=p)
        # create a list with samples per label for the needed names
        samples_per_labels = [samples_per_names[name] for name in names if name in names]
        for samples_per_label in samples_per_labels:
            # remove patient index
            samples_per_label.pop(FEATURE_PAT_IDX)
            yield samples_per_label


def _get_samples_per_name(file_path: str) -> dict:
    # open meta file
    info = json.load(open(file=file_path, mode="r"))
    # get the section with the samples per name
    return info[SAMPLES_PER_NAME]
