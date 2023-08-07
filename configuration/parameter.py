ZARR_PAT_DATA = "patients_data.zarr"    # Archive name for zarr
PAT_CHUNKS = (1000,)
D3_PAT_CHUNKS = PAT_CHUNKS + (1, 1)

# --- Data archive names
DICT_X = "X"                            # default dict name for spectrum
DICT_y = "y"                            # default dict name for classifications
DICT_IDX = "indexes_in_datacube"        # default dict name for original indexes in hyper cube
DICT_WEIGHT = "weights"                 # default dict name for weights
