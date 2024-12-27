from pathlib import Path


# ----------------------- Project's directory structure ---------------------- #

# Root dir
ROOT_DIR = Path(__file__).parent.parent.parent

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
MESH_DROP_PERCENTAGE = 0.85

# ----------------------------------- Model ---------------------------------- #
# LAYERS_IN = [26404, 10000, 5000, 2500]  # 26404 being the size of the meshes in our dataset
# LATENT_SIZE = 512
# TEXT_EMBED_SIZE = 512

LAYERS_IN = [24, 10]  # 26404 being the size of the meshes in our dataset
LATENT_SIZE = 3
TEXT_EMBED_SIZE = 3

# --------------------------------- Training --------------------------------- #

BATCH_SIZE = 4
EPOCHS = 50
PATIENCE = 20