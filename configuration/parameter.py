ARCHIVE_TYPE = "npz"                # type for data archive management

# --- ZARR params
BLOCK_SIZE = 10000                  # max number of samples per chunk
D3_CHUNK = (1, 1)                   # chunks for patches

# --- shuffle
SHUFFLE_GROUP_NAME = "shuffled"     # prefix for shuffled dataset
PILE_NAME = ".pile"                 # data extension for piles
MAX_SIZE_PER_PILE = 2.0             # maximal size per pile/shuffle file in GB

# --- TFR meta files
SAMPLES_PER_NAME = "samples_per_patient_name"
TOTAL_SAMPLES = "total_samples"
TFR_META_EXTENSION = ".tfrmeta"


# --- Data archive names
DICT_X = "X"                            # default dict name for spectrum
DICT_y = "y"                            # default dict name for classifications
DICT_IDX = "indexes_in_datacube"        # default dict name for original indexes in hyper cube
DICT_WEIGHT = "weights"                 # default dict name for weights
ORG_NAME = "org_name"                   # original file name, when mor files in one data archive

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

# --- TFRecord Dataset
FEATURE_X = "X"                         # name for TFRecord feature with X list
FEATURE_Y = "y"                         # name for TFRecord feature with y list
FEATURE_SAMPLES = "samples"             # name for TFRecord feature with samples count (X axis 0, y axis 0)
FEATURE_SPEC = "spectrum"               # name for TFRecord feature with spectrum count (X axis last)
FEATURE_X_AXIS_1 = "X_patch_0"          # name for TFRecord feature with X axis 1 size
FEATURE_X_AXIS_2 = "X_patch_1"          # name for TFRecord feature with X axis 2 size
FEATURE_WEIGHTS = "sample_weights"      # name for TFRecord feature with sample weight list
FEATURE_IDX_CUBE = "indexes_in_datacube"    # name for TFRecord feature withe original index in datacube
FEATURE_PAT_IDX = "patient_index"       # name for TFRecord feature withe patient index
FEATURE_PAT_NAME = "patient_names"       # name for TFRecord feature withe patient name
PAT_NAME_SEPERATOR = ","
PAT_NAME_ENCODING = "UTF-8"
TFR_FILE_EXTENSION = ".tfrecord"

# --- Generator
GEN_ALL = "all"                         # modus all for data generator
GEN_TRAIN = "train"                     # modus train for data generator
GEN_VALID = "valid"                     # modus valid for data generator

# --- prediction to image
PRED_TO_IMG_TYP = "archive"
