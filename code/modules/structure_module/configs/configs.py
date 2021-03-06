import os
import logging

SA_HOME = os.path.join(os.getcwd(), "modules/structure_module")
# dependency location and location to store temporary files running the dependencies
ALGO_BASE_DIRS = {
    "JDC": f"{SA_HOME}/third_party/melodyExtraction_JDC",
    "SSL": f"{SA_HOME}/third_party/melodyExtraction_SSL",
    "PopMusicHighlighter": f"{SA_HOME}/third_party/pop-music-highlighter",
    "TmpDir": f"{SA_HOME}/tmp/MIR",
}
# dataset location and preprocess cache files location
DATASET_BASE_DIRS = {
    "SALAMI": f"{SA_HOME}/dataset/salami",
    "RWC": f"{SA_HOME}/dataset/RWC",
    "RWC_accomp": f"{SA_HOME}/dataset/RWC-accompaniment",
    "CCM": f"{SA_HOME}/dataset/CCM_Structure",
    "Huawei": f"{SA_HOME}/dataset/Huawei",
    "LocalTemporary_Dataset": f"{SA_HOME}/dataset/localTmp",
}
# output data location
EVAL_RESULT_DIR = f"{SA_HOME}/data/evalResult/"
MODELS_DIR = f"{SA_HOME}/data/models"
VIEWER_DATA_DIR = f"{SA_HOME}/data/viewerMetadata"
PRED_DIR = f"{SA_HOME}/data/predict"

# evaluation settings
FORCE_EVAL = False
METRIC_NAMES = [
    "ovlp-P",
    "ovlp-R",
    "ovlp-F",
    "sovl-P",
    "sovl-R",
    "sovl-F",
    "dtct-P",
    "dtct-R",
    "dtct-F",
]
PLOT_METRIC_FIELDS = [
    ["ovlp-F"],
    ["sovl-F"],
    # ["dtct-P"],
    # ["dtct-R"],
    # ["dtct-F"],
]
# PLOT_METRIC_FIELDS = [
#     ['ovlp-P', 'ovlp-R', 'ovlp-F'],
#     ['sovl-P', 'sovl-R', 'sovl-F'],
# ]
# PLOT_METRIC_FIELDS = [
#     ['ovlp-P', 'ovlp-R'],
#     ['ovlp-F', 'sovl-F'],
# ]
DETECTION_WINDOW = 3

# logging settings
DEBUG = True if os.getenv("DEBUG") is not None else False
logger = logging.getLogger("chorus_detector")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG) if DEBUG else ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# make these directories
mk_dirs = [
    EVAL_RESULT_DIR,
    MODELS_DIR,
    VIEWER_DATA_DIR,
    PRED_DIR,
    ALGO_BASE_DIRS["TmpDir"],
    DATASET_BASE_DIRS["LocalTemporary_Dataset"],
]
for path in mk_dirs:
    if not os.path.exists(path):
        dirname = os.path.dirname(path)
        if os.path.exists(dirname):
            os.mkdir(path)
        else:
            logger.warn(f"directory={dirname} does not exist")

# process numbers for parallel computing
NUM_WORKERS = os.cpu_count() // 2 if not DEBUG else 1
