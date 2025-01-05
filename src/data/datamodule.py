from pathlib import Path
import torch
from torch.utils.data import Dataset, default_collate
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import glob
from typing import Optional, Literal
from src.utils.mesh_utils import read_mesh
from src.utils.file_utils import download_resource
from src.utils.file_utils import download_google_drive
from src.utils.file_utils import extract_zip
from src.utils.file_utils import remove


def collate_meshes(batch):
    """Custom collate function for batching meshes together."""
    # Extract components from each mesh
    verts = [item["verts"] for item in batch]
    faces = [item["faces"] for item in batch]
    descriptions = [item["description"] for item in batch]

    # Create batched meshes object
    meshes = Meshes(
        verts=verts,
        faces=faces,
    )

    return {
        "mesh": meshes,
        "description": descriptions
    }


class FacescapeDataset(Dataset):
    """
    A PyTorch Dataset class for handling Facescape data.

    Args:
        data (list): The dataset.
        mesh_drop_percent (float): Percentage of faces to drop (between 0.0 and 1.0), used if face_count is None
        mesh_face_count (int, optional): Target number of faces in simplified mesh, overrides percent if provided
        aggression (int): Simplification aggressiveness, 0 (slow/quality) to 10 (fast/rough)
    """

    def __init__(
            self,
            data: Path,
            mesh_drop_percent: float = None,
            mesh_face_count: Optional[int] = None,
            aggression: Optional[int] = None,
            loader: Literal['pytorch3d', 'trimesh'] = 'pytorch3d',
            normalize: bool = False
    ):
        """
        Args:
            data_dir: Root directory containing mesh files
            mesh_drop_percent: Percentage of faces to drop (between 0.0 and 1.0)
            mesh_face_count: Target number of faces in simplified mesh
            aggression: Simplification aggressiveness (0-10)
            loader: Mesh loader to use ('pytorch3d' or 'trimesh')
            normalize: Whether to normalize vertex coordinates
        """
        self.data = data
        self.mesh_drop_percent = mesh_drop_percent
        self.mesh_face_count = mesh_face_count
        self.aggression = aggression
        self.loader = loader
        self.normalize = normalize

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            item (int): The index of the desired item.

        Returns:
            dict: A dictionary containing the item data.
        """
        item = self.data[idx]

        # Load and process mesh
        neutral_mesh = read_mesh(
            str(item['neutral_path']),
            loader=self.loader,
            mesh_drop_percent=self.mesh_drop_percent,
            mesh_face_count=self.mesh_face_count,
            aggression=self.aggression,
            normalize=self.normalize,
        )

        expression_mesh = read_mesh(
            str(item['expression_path']),
            loader=self.loader,
            mesh_drop_percent=self.mesh_drop_percent,
            mesh_face_count=self.mesh_face_count,
            aggression=self.aggression,
            normalize=self.normalize,
        )

        return {
            'neutral_graph': neutral_mesh,
            'expression_graph': expression_mesh,
            'description': item['description']
        }


class FacescapeDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for 3D face mesh data."""

    def __init__(
            self,
            resource_url: str,
            download_source: str,
            data_dir: Path,
            download: str = 'infer',
            text_generation=None,
            batch_size: int = 32,
            num_workers: int = 4,
            train_split: float = 0.8,
            val_split: float = 0.1,
            mesh_drop_percent: float = None,
            mesh_face_count: Optional[int] = None,
            aggression: Optional[int] = None,
            loader: Literal['pytorch3d', 'trimesh'] = 'pytorch3d',
            normalize: bool = False,
            custom_collate=default_collate
    ):
        """
        Args:
            resource_url: URL for downloading the dataset
            download_source: Type of download source ('drive' or 'url')
            data_dir: Directory where the data will be stored
            download: Download behavior ('infer', 'yes', or 'no')
            text_generation: Function to generate descriptions from filenames
            batch_size: Batch size for dataloaders
            num_workers: Number of CPU workers for data loading
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            mesh_pattern: Pattern to match mesh files
            mesh_drop_percent: Percentage of faces to drop
            mesh_face_count: Target number of faces in simplified mesh
            aggression: Simplification aggressiveness (0-10)
            loader: Mesh loader to use ('pytorch3d' or 'trimesh')
            normalize: Whether to normalize vertex coordinates
        """
        super().__init__()
        self.resource_url = resource_url
        self.download_source = download_source
        self.data_dir = data_dir
        self.download = download

        if text_generation is None:
            text_generation = lambda s: ''.join(c if not c.isdigit() else '' for c in s.replace('_', ' '))

        self.text_generation = text_generation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.mesh_drop_percent = mesh_drop_percent
        self.mesh_face_count = mesh_face_count
        self.aggression = aggression
        self.loader = loader
        self.normalize = normalize
        self.custom_collate = custom_collate

        # Default required files structure
        self.required_files = [
            Path(self.data_dir, f'{i}', 'models_reg') for i in range(100)
        ]

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
            # Check if all the required paths exists and, if a path points to a directory,
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
                raise ValueError(f"Unsupported download source: {self.download_source}")

            print('Resource downloaded. Extracting...')
            extract_zip(zip_path, self.data_dir)
            print('Resource extracted. Removing zip file...')
            remove(zip_path)
            print('Done.')
        else:
            print('Resource already downloaded.')

        self.data = []
        for user_folder in self.data_dir.iterdir():
            user_path = Path(self.data_dir, user_folder, 'models_reg')

            if user_path not in self.required_files:
                continue

            if user_path.is_dir():
                # Find the path to the neutral mesh
                neutral_path = None
                for file in user_path.iterdir():
                    if "neutral" in file.stem and file.suffix == ".obj":
                        neutral_path = Path(user_path, file)
                        break

                if not neutral_path:
                    print(f"No neutral mesh found for user {user_folder}. Skipping...")
                    continue

                # Process all other meshes
                for file in user_path.iterdir():
                    if file.suffix == ".obj" and file != neutral_path:
                        description = self.text_generation(file.stem)
                        self.data.append(
                            {
                                'neutral_path': neutral_path,
                                'expression_path': file,
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

        # Create the dataset with simplification parameters
        full_dataset = FacescapeDataset(
            self.data,
            mesh_drop_percent=self.mesh_drop_percent,
            mesh_face_count=self.mesh_face_count,
            aggression=self.aggression,
            loader=self.loader,
            normalize=self.normalize
        )

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
            num_workers=self.num_workers,
            shuffle=True
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

    from src.models.gnn import config
    from src.data.text_generation import DEFAULT_TEXT_GENERATION

    from src.utils.mesh_utils import visualize_mesh

    datamodule = FacescapeDataModule(
        resource_url=config.RESOURCE_URL,
        download_source=config.DOWNLOAD_SOURCE,
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        download=config.DOWNLOAD,
        text_generation=DEFAULT_TEXT_GENERATION,
        custom_collate=collate_meshes
    )

    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()

    iter_dataloader = iter(train_dataloader)
    for batch in iter_dataloader:

        print(f'\nNumber of samples in the first batch: {len(list(batch.values())[0])}')

        print(f'\nBatch consist of:')
        for key, value in batch.items():
            print(f'{key}: {value}')

        print(f'\nIterating batch...')
        neutral_databatch = batch['neutral_graph']

        break
