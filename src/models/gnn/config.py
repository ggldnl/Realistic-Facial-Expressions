from pathlib import Path


# ----------------------- Project's directory structure ---------------------- #

# Root dir
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Where to check or put the downloaded data
DATA_DIR = Path(ROOT_DIR, 'datasets/facescape_highlighted')

# Model folder
MODEL_DIR = Path(ROOT_DIR, 'src/models/gnn')

# Where to put renderings and logs
RENDER_DIR = Path(MODEL_DIR, 'data/renders')
LOG_DIR = Path(MODEL_DIR, 'data/logs')

# ------------------------------- Preprocessing ------------------------------ #

MESH_DROP_PERCENTAGE = 0.8  # Drop 80% of the mesh

# ---------------------------------- Dataset --------------------------------- #

# URL that identifies where to download the data from
# RESOURCE_URL = r"https://drive.google.com/file/d/1owzGMup14KBclXstg4dQh91PfQo2kDsV/view?usp=sharing"  # 1-100 preprocessed
RESOURCE_URL = r"https://drive.google.com/file/d/1Ajk8hCGI_bcT5dYmb4H1Srw1eOJ-ykaa/view?usp=sharing"  # 100

# The URL can point to google drive, author's website and so on
DOWNLOAD_SOURCE = 'drive'

# Download, skip or check if all the files are there before downloading
DOWNLOAD = 'infer'

FIRST_SUBJECT_INDEX = 1
LAST_SUBJECT_INDEX = 100

# Splitting percentage
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.1
# 0.1 percent will be devoted to the validation set

# ----------------------------------- Model ---------------------------------- #

INPUT_DIM = 3
LATENT_SIZE = 3

# --------------------------------- Training --------------------------------- #

BATCH_SIZE = 2
EPOCHS = 50
PATIENCE = 20
LEARNING_RATE = 1e-3
RENDER_INTERVAL = 1
RENDER_PROMPT = 'eye closed'
RENDER_IN = DATA_DIR / '100/models_reg/1_neutral.obj'
RENDER_REF = DATA_DIR / '100/models_reg/18_eye_closed.obj'
RENDER_RADIUS = 5

# --------------------------------- Loss Weights ---------------------------- #

# Loss function parameters
W_CHAMFER = 1     # Weight for chamfer loss
W_NORMAL = 1       # Weight for normal consistency loss
W_LAPLACIAN = 1    # Weight for laplacian smoothing loss
N_SAMPLES = 5000     # Number of points to sample for chamfer loss
