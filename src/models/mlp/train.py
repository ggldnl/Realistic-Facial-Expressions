from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.mlp.model import NeuralStyleTransfer

from src.config import config
from src.data.datamodule import FacescapeDataModule
from src.data.text_generation import DEFAULT_TEXT_GENERATION


def train_style_transfer(
        output_dir: str,
        prompt: Optional[str] = None,
        reference_image: Optional[str] = None,
        max_epochs: int = 1000,
        **kwargs
):
    """Main training function"""

    # Set up data module
    dm =FacescapeDataModule(
        resource_url=config.RESOURCE_URL,
        download_source=config.DOWNLOAD_SOURCE,
        data_dir=config.DATA_DIR,
        download=config.DOWNLOAD,
        text_generation=DEFAULT_TEXT_GENERATION
    )

    # Initialize model
    model = NeuralStyleTransfer(
        prompt=prompt,
        reference_image=reference_image,
        **kwargs
    )

    # Configure trainer
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            ModelCheckpoint(
                monitor='train_loss',
                dirpath=output_dir,
                filename='model-{epoch:02d}-{train_loss:.2f}',
                save_top_k=3
            ),
            LearningRateMonitor(logging_interval='epoch')
        ],
        logger=TensorBoardLogger(output_dir, name='neural_style')
    )

    # Prepare and train
    dm.prepare_data()
    dm.setup()
    trainer.fit(model, dm)

if __name__ == '__main__':
    train_style_transfer(
        data_dir=config.ROOT_DIR / "output",
        output_dir='output',
        prompt='your style prompt',
        max_epochs=1000
    )