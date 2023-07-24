from configuration.configloader_dataloader import read_dataloader_config


GET_DATALOADER_DATA = {"TYPE": "normal",
                       "FILE_EXTENSION": ".dat",
                       "3D": True,
                       "3D_SIZE": [3, 3],
                       "FIRST_NM": 8,
                       "LAST_NM": 100,
                       "WAVE_AREA": 100,
                       "LABELS_TO_TRAIN": [0, 1],
                       "NAME_SPLIT": "_SpecCube",
                       "MASK_DIFF": ["_SpecCube.dat", ".png"],
                       "LABELS_FILENAME": "labels.labels",
                       "CONTAMINATION_FILENAME": "contamination.csv",
                       "SMOOTHING_TYPE": "median_filter",
                       "SMOOTHING_VALUE": 5,
                       "BORDER_CONFIG": {
                           "enable": False,
                           "methode": "detect_core",
                           "depth": 5,
                           "axis": [],
                           "not_used_labels": []
                       },
                       "SPLIT_PATHS_BY": "Files",
                       "CV_HOW_MANY_PATIENTS_EXCLUDE_FOR_TEST": 1,
                       "WITH_BACKGROUND_EXTRACTION": False,
                       "MASK_COLOR": {0: [[255, 255, 0]], 1: [[0, 0, 255]], 2: [[255, 0, 0]]},
                       "TISSUE_LABELS": {0: "Nerve", 1: "Tumor", 2: "Parotis"},
                       "PLOT_COLORS": {0: "yellow", 1: "blue", 2: "red"},
                       "LABELS": [0, 1, 2]}


def test_get_dataloader(dataloader_data_dir):
    assert read_dataloader_config(file=dataloader_data_dir, section="DATALOADER") == GET_DATALOADER_DATA
    