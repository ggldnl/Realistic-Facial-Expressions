from typing import Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import config
from model import NeuralStyleTransfer

from src.data.datamodule import FacescapeDataModule
from src.data.text_generation import DEFAULT_TEXT_GENERATION


def train_style_transfer(
        output_dir: str,
        max_epochs: int = 6000,
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
        batch_size=config.BATCH_SIZE,
        **kwargs
    )

    # Configure trainer
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        accelerator='cpu',
        callbacks=[
            ModelCheckpoint(
                monitor='train_loss',
                dirpath=output_dir,
                filename='model-{epoch:02d}-{train_loss:.2f}',
                save_top_k=3
            ),
            LearningRateMonitor(logging_interval='epoch')
        ],
        #logger=TensorBoardLogger(output_dir, name='neural_style')
    )

    # Prepare and train
    #dm.prepare_data()
    #dm.setup()
    trainer.fit(model, dm)

if __name__ == '__main__':
    train_style_transfer(
        output_dir=config.DATA_DIR / 'output',
        max_epochs=10000
    )