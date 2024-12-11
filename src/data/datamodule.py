import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import os

from src.utils.mesh_utils import read_mesh
from src.utils.file_utils import download_resource
from src.utils.file_utils import download_google_drive
from src.utils.file_utils import extract_zip
from src.utils.file_utils import remove


seed = 127001
rng = np.random.default_rng(seed)


class FacescapeDataset(Dataset):
    """
    A PyTorch Dataset class for handling Facescape data.

    Args:
        data (list): The dataset.
    """

    def __init__(self, data):
        super(FacescapeDataset, self).__init__()

        self.data = data

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        Retrieves an item from the dataset.

        Args:
            item (int): The index of the desired item.

        Returns:
            dict: A dictionary containing the item data.
        """

        return {
            'neutral': read_mesh(self.data[item]['neutral']),
            'expression': read_mesh(self.data[item]['expression']),
            'description': self.data[item]['description']  # TODO add text encoder
        }


class FacescapeDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for handling data loading, downloading, and preprocessing of the
    Facescape dataset.

    Args:
        resource_url (str): URL for downloading the dataset.
        download_source (str): Type of download source ('drive' or 'url').
        data_dir (Path): Directory where the data will be stored.
        download (str): Behavior for downloading ('infer', 'yes', or 'no').
        batch_size (int): Batch size for DataLoaders.
        custom_collate (callable, optional): Custom collate function for DataLoaders.
        num_workers (int): Number of workers for DataLoaders.
        train_split (float): Proportion of data for training.
        val_split (float): Proportion of data for validation.
    """

    def __init__(self,
                 resource_url,          # URL that identifies where to download the data from
                 download_source,       # The URL can point to google drive, author's website and so on
                 data_dir,              # Where to put the downloaded data
                 download='infer',      # Download, skip or check if all the files are there before downloading
                 text_generation=None,  # Lambda used to generate a text description given a file name
                 batch_size=64,
                 custom_collate=None,
                 num_workers=4,
                 train_split=0.8,
                 val_split=0.1,
                 ):

        super().__init__()

        # Data downloading
        self.resource_url = resource_url
        self.download_source = download_source
        self.data_dir = data_dir
        self.download = download

        if text_generation is None:

            # Lambda to:
            # 1. replace "_" with " "
            # 2. remove digits
            text_generation = lambda s: ''.join(c if not c.isdigit() else '' for c in s.replace('_', ' '))

        self.text_generation = text_generation

        # TODO for now, only one user is taken into account
        self.required_files = [
            Path(self.data_dir, f'{i}') for i in [200]
        ]

        # Other stuff
        self.batch_size = batch_size
        self.custom_collate = custom_collate
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split

        self.data = None
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """
        Prepares the dataset by downloading and extracting it if necessary.
        Also performs initial setup like creating the data directory.
        """
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.download == 'infer':
            # Check if all the required paths exist and, if a path points to a directory,
            # check if it contains at least one element
            if any(not path.exists() or (path.is_dir() and not any(path.iterdir())) for path in self.required_files):
                self.download = 'yes'

        if self.download == 'yes':
            zip_path = Path(self.data_dir, 'datamodule.zip')
            print('Downloading resource...')

            if self.download_source == 'drive':
                download_google_drive(self.resource_url, zip_path)
            elif self.download_source == 'url':
                download_resource(self.resource_url, zip_path)
            else:
                raise ValueError(f"Unsupported datamodule source: {self.download_source}")
            print('Resource downloaded.\nExtracting resource...')

            extract_zip(zip_path, self.data_dir)
            print('Resource extracted.\nRemoving zip file...')
            remove(zip_path)
            print('Done.')
        else:
            print('Resource already downloaded.')

        self.data = []
        for user_folder in self.data_dir.iterdir():
            user_path = Path(self.data_dir, user_folder)

            if user_path not in self.required_files:
                continue

            if user_path.is_dir():
                # Find the path to the neutral mesh
                neutral_mesh = None
                for file in user_path.iterdir():
                    if "neutral" in file.stem and file.suffix == ".ply":
                        neutral_mesh = Path(user_path, file)
                        break

                if not neutral_mesh:
                    print(f"No neutral mesh found for user {user_folder}. Skipping...")
                    continue

                # Process all other meshes
                for file in user_path.iterdir():
                    if file.suffix == ".ply" and file != neutral_mesh:
                        description = self.text_generation(file.stem)
                        self.data.append(
                            {
                                'neutral': neutral_mesh,
                                'expression': file,
                                'description': description
                            }
                        )

        print('Data acquired.')

    def setup(self, stage=None):
        """
        Prepares the data for training, validation, and testing.

        Args:
            stage (str, optional): The current stage of training ('fit', 'validate', or 'test').
        """
        if self.data is None:
            raise ValueError('You should prepare the data first (calling prepare_data())')

        # Create the dataset (store it once computed)
        full_dataset = FacescapeDataset(self.data)

        # Compute the sizes for train, val, and test splits
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size

        # Split the dataset (this keeps the flags of the big dataset)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        """
        Creates a DataLoader for the training set.

        Returns:
            DataLoader: A DataLoader for the training set.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.custom_collate,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        """
        Creates a DataLoader for the validation set.

        Returns:
            DataLoader: A DataLoader for the validation set.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.custom_collate,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        """
        Creates a DataLoader for the test set.

        Returns:
            DataLoader: A DataLoader for the test set.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.custom_collate,
            num_workers=self.num_workers
        )


if __name__ == '__main__':

    import src.config.config as config
    from src.data.text_generation import DEFAULT_TEXT_GENERATION

    datamodule = FacescapeDataModule(
        resource_url=config.RESOURCE_URL,
        download_source=config.DOWNLOAD_SOURCE,
        data_dir=config.DATA_DIR,
        download=config.DOWNLOAD,
        text_generation=DEFAULT_TEXT_GENERATION
    )

    datamodule.prepare_data()
    datamodule.setup()

    # Get a datamodule to iterate
    train_loader = datamodule.train_dataloader()
    print(f'Number of batches in the train dataset: {len(train_loader)}')

    for elem in train_loader:
        print(f'Number of samples in the train dataset: {len(list(elem.values())[0])}')
        for key, value in elem.items():
            print(f'{key}: {value}')
        break
