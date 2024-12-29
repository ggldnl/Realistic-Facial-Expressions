from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from src.data.text_generation import DEFAULT_TEXT_GENERATION

from src.data.datamodule import FacescapeDataModule
from src.models.gnn.model import Model
import src.models.gnn.config as config
from src.models.gnn.callbacks import RenderCallback


if __name__ == '__main__':

    # Create a datamodule
    datamodule = FacescapeDataModule(
        resource_url=config.RESOURCE_URL,
        download_source=config.DOWNLOAD_SOURCE,
        data_dir=config.DATA_DIR,
        download=config.DOWNLOAD,
        text_generation=DEFAULT_TEXT_GENERATION,
        batch_size=config.BATCH_SIZE,
        mesh_drop_percent=config.MESH_DROP_PERCENTAGE
    )

    # Log to tensorboard
    logger = TensorBoardLogger(config.LOG_DIR, name="GNN")

    # Create the model
    model = Model(
        config.LATENT_SIZE,
        lr=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE
    )

    render_callback = RenderCallback(
        n_epochs=config.RENDER_INTERVAL,
        model=model,
        in_mesh=config.DATA_DIR / '100/models_reg/1_neutral.obj',
        out_dir=config.RENDER_DIR,
        prompt=config.RENDER_PROMPT
    )

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        logger=logger,
        accelerator='auto',
        # gradient_clip_val=1.0,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            EarlyStopping(monitor='val_loss', patience=config.PATIENCE, mode='min'),

            # Custom render callback
            render_callback
        ]
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
