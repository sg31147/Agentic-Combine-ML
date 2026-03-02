from pathlib import Path

from omegaconf import OmegaConf

PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"

ID_COLUMN = "_id"
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
SUBJECT_ID_COLUMN = "subject_id"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOWNLOAD_DIRECTORY_MIMICIV = str(PROJECT_ROOT / "dataset" / "mimiciv")  # Path to the MIMIC-IV data. Example: ~/physionet.org/files/mimiciv/2.2
DOWNLOAD_DIRECTORY_MIMICIV_NOTE = str(PROJECT_ROOT / "dataset" / "mimiciv")  # Path to the MIMIC-IV-Note data. Example: ~/physionet.org/files/mimic-iv-note/2.2



DATA_DIRECTORY_MIMICIV_ICD10 = OmegaConf.load(
    PROJECT_ROOT / "configs" / "data" / "mimiciv_icd10.yaml"
).dir

# this variable is used for genersating plots and tables from wandb

PROJECT = "pseudo-relevance feedback" 

EXPERIMENT_DIR = str(PROJECT_ROOT / "experiments")  # Path to the experiment directory. Example: ~/experiments
PALETTE = {
    "PLM-ICD": "#E69F00",
    "LAAT": "#009E73",
    "MultiResCNN": "#D55E00",
    "CAML": "#56B4E9",
    "CNN": "#CC79A7",
    "Bi-GRU": "#F5C710",
}
HUE_ORDER = ["PLM-ICD", "LAAT", "MultiResCNN", "CAML", "Bi-GRU", "CNN"]
MODEL_NAMES = {"PLMICD": "PLM-ICD", "VanillaConv": "CNN", "VanillaRNN": "Bi-GRU"}


best_runs = {
    "CAML": str(PROJECT_ROOT / "experiments" / "xsm4ojqd"),
    "LAAT": str(PROJECT_ROOT / "experiments" / "rep5wxro"),
    "PLMICD": str(PROJECT_ROOT / "experiments" / "8fdtxcm4"),
    "MultiResCNN": str(PROJECT_ROOT / "experiments" / "jc1u3c6s"),
    "VanillaRNN": str(PROJECT_ROOT / "experiments" / "djhscsu3"),
    "VanillaConv": str(PROJECT_ROOT / "experiments" / "kr2vh2uf"),
}
