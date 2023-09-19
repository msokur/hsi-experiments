ARCHIVE_TYPE = "zarr"                # type for data archive management

# --- ZARR params
PAT_CHUNKS = (1000,)
D3_PAT_CHUNKS = PAT_CHUNKS + (1, 1)

# --- shuffle
SHUFFLE_GROUP_NAME = "shuffled"     # prefix for shuffled dataset
PILE_NAME = ".pile"                 # data extension for piles


# --- Data archive names
DICT_X = "X"                            # default dict name for spectrum
DICT_y = "y"                            # default dict name for classifications
DICT_IDX = "indexes_in_datacube"        # default dict name for original indexes in hyper cube
DICT_WEIGHT = "weights"                 # default dict name for weights

# --- Scaler
SCALER_FILE = "scaler.scaler"

# --- Trainer
VALID_LOG = "valid.valid_except_names"  # data name to log valid names
HISTORY_FILE = "history"                # file name for training history

# --- batches
TUNE = "tune"                           # add to batchfolder for tune dataset
TRAIN = "train"                         # key for train dataset
VALID = "valid"                         # key for valid dataset
BATCH_FILE = "batch"                    # prefix for batch file
BATCH_IDX = "batch_indexes"             # name for array with
BATCH_ORG_PATH = "batch_org_path"       # name for array with original data for batches (zarr)
MODEL_BATCH_SIZE = 500

# --- Generator
GEN_ALL = "all"                         # modus all for data generator
GEN_TRAIN = "train"                     # modus train for data generator
GEN_VALID = "valid"                     # modus valid for data generator

# --- prediction to image
PRED_TO_IMG_TYP = "archive"