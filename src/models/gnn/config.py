from pathlib import Path


# ----------------------- Project's directory structure ---------------------- #

# Root dir
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Where to check or put the downloaded data
DATA_DIR = Path(ROOT_DIR, 'datasets/simplified_meshes')

# Model folder
MODEL_DIR = Path(ROOT_DIR, 'src/models/gnn')

# Where to put renderings and logs
RENDER_DIR = Path(MODEL_DIR, 'data/renders')
LOG_DIR = Path(MODEL_DIR, 'data/logs')

# ---------------------------------- Dataset --------------------------------- #

# URL that identifies where to download the data from
RESOURCE_URL = r"https://drive.google.com/drive/folders/14aQeK7S35FP5Z9EkJ6e6TCZLQxCYRJZE?usp=drive_link"

# The URL can point to google drive, author's website and so on
DOWNLOAD_SOURCE = 'drive'

# Download, skip or check if all the files are there before downloading
DOWNLOAD = 'infer'

# Splitting percentage
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.1
# 0.1 percent will be devoted to the validation set

MESH_DROP_PERCENTAGE = 0.8  # Drop 80% of the mesh

# ----------------------------------- Model ---------------------------------- #
LATENT_SIZE = 512

# --------------------------------- Training --------------------------------- #

BATCH_SIZE = 2
EPOCHS = 50
PATIENCE = 20
LEARNING_RATE = 1e-3
RENDER_INTERVAL = 1
RENDER_PROMPT = 'smile'
RENDER_RADIUS = 5
