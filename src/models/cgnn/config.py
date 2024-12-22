from pathlib import Path


# ----------------------- Project's directory structure ---------------------- #

# Root dir
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Where to check or put the downloaded data
DATA_DIR = Path(ROOT_DIR, 'datasets/facescape')

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

# mesh simplification
MESH_SIMPLIFY = True
MESH_DROP_PERCENTAGE = 0.8

# ----------------------------------- Model ---------------------------------- #
LATENT_SIZE = 512

# --------------------------------- Training --------------------------------- #

BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 20
LEARNING_RATE = 1e-3
