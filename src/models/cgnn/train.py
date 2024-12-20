from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from src.data.text_generation import DEFAULT_TEXT_GENERATION
from src.data.datamodule import FacescapeDataModule
from src.models.cgnn.model import Model
import src.models.cgnn.config as config


if __name__ == '__main__':

    # Create a datamodule
    datamodule = FacescapeDataModule(
        resource_url=config.RESOURCE_URL,
        download_source=config.DOWNLOAD_SOURCE,
        data_dir=config.DATA_DIR,
        download=config.DOWNLOAD,
        text_generation=DEFAULT_TEXT_GENERATION,
        batch_size=config.BATCH_SIZE
    )

    # Log to tensorboard
    logger = TensorBoardLogger("data/logs/", name="ConvGNN")

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        logger=logger,
        # gradient_clip_val=1.0,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            EarlyStopping(monitor='val_loss', patience=config.PATIENCE, mode='min')
        ]
    )

    # Create the model
    model = Model(
        config.LATENT_SIZE,
        lr=config.LEARNING_RATE
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
