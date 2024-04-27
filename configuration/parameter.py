STORAGE_TYPE = "npz"                # type for data archive management choose between 'npz' or 'zarr'
DATASET_TYPE = "generator"          # type of dataset, choose between 'generator' or 'tfr'

# --- ZARR params
BLOCK_SIZE = 10000                  # max number of samples per chunk
D3_CHUNK = (1, 1)                   # chunks for patches

# --- shuffle
SHUFFLE_GROUP_NAME = "shuffled"     # prefix for shuffled dataset
PILE_NAME = ".pile"                 # data extension for piles
MAX_SIZE_PER_PILE = 2.0             # maximal size per pile/shuffle file in GB
SMALL_SHUFFLE_SET = False           # create only representative shuffle sets

# --- Meta files
SAMPLES_PER_NAME = "samples_per_patient_name"
TOTAL_SAMPLES = "total_samples"
TFR_TYP = "tfr"                     # key for tfr meta file
GEN_TYP = "generator"               # key for generator meta file
TFR_META_EXTENSION = ".tfrmeta"     # extension for tfr meta file
GEN_META_EXTENSION = ".meta"        # extension for generator dataset meta file


# --- Data storage names
DICT_X = "X"                            # default dict name for spectrum
DICT_y = "y"                            # default dict name for classifications
DICT_IDX = "indexes_in_datacube"        # default dict name for original indexes in hyper cube
DICT_WEIGHT = "weights"                 # default dict name for weights
ORIGINAL_NAME = "original_name"                   # original file name, when mor files in one data storage
BACKGROUND_MASK = 'background_mask'

# --- Scaler
SCALER_FILE = "scaler.scaler"

# --- Trainer
FILE_WITH_VALID_NAME = "valid.valid_except_names"  # data name to log valid names
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

# --- predictor
MAX_SIZE_PER_SPEC = 4.0                 # maximal size in GB for a spectrum to predict

# --- prediction to image
PRED_TO_IMG_TYP = "archive"
