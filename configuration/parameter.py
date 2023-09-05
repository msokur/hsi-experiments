ARCHIVE_TYPE = "zarr"

# --- ZARR params
PAT_CHUNKS = (1000,)
D3_PAT_CHUNKS = PAT_CHUNKS + (1, 1)

SHUFFLE_ARCHIVE = "shuffled_data"    # Archive name for shuffled zarr

SHUFFLE_GROUP_NAME = "shuffled"
PILE_NAME = ".pile"


# --- Data archive names
DICT_X = "X"                            # default dict name for spectrum
DICT_y = "y"                            # default dict name for classifications
DICT_IDX = "indexes_in_datacube"        # default dict name for original indexes in hyper cube
DICT_WEIGHT = "weights"                 # default dict name for weights

# --- Scaler
SCALER_FILE = "scaler.scaler"
