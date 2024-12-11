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