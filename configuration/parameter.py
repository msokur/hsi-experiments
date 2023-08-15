ZARR_PAT_GROUP = "patients_data.zarr"    # Archive name for zarr
PAT_CHUNKS = (1000,)
D3_PAT_CHUNKS = PAT_CHUNKS + (1, 1)

ZARR_SHUFFLE_GROUP = "shuffled_data.zarr"    # Archive name for shuffled zarr
ZARR_SHUFFLE_NAME_ARRAY = "pat_name"
ZARR_SHUFFLE_IDX_ARRAY = "data_idx"


# --- Data archive names
DICT_X = "X"                            # default dict name for spectrum
DICT_y = "y"                            # default dict name for classifications
DICT_IDX = "indexes_in_datacube"        # default dict name for original indexes in hyper cube
DICT_WEIGHT = "weights"                 # default dict name for weights

# --- Scaler
SCALER_FILE = "scaler.scaler"
