from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from src.data.text_generation import DEFAULT_TEXT_GENERATION

from src.data.datamodule import FacescapeDataModule
from src.models.gnn.model import Model
import src.models.gnn.config as config
from src.utils.callbacks import RenderCallback
from src.utils.renderer import Renderer

if __name__ == '__main__':

    # Create a datamodule
    datamodule = FacescapeDataModule(
        resource_url=config.RESOURCE_URL,
        download_source=config.DOWNLOAD_SOURCE,
        data_dir=config.DATA_DIR,
        download=config.DOWNLOAD,
        text_generation=DEFAULT_TEXT_GENERATION,
        batch_size=config.BATCH_SIZE,
    )

    # Log to tensorboard
    logger = TensorBoardLogger(config.LOG_DIR, name="GNN")

    # Create the model
    model = Model(
        latent_size=config.LATENT_SIZE,
        input_dim=config.INPUT_DIM,
        lr=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        w_chamfer=config.W_CHAMFER,
        w_normal=config.W_NORMAL,
        w_laplacian=config.W_LAPLACIAN,
        n_samples=config.N_SAMPLES
    )

    render_callback = RenderCallback(
        n_epochs=config.RENDER_INTERVAL,
        model=model,
        renderer=Renderer(),
        in_mesh=config.RENDER_IN,
        out_dir=config.RENDER_DIR,
        prompt=config.RENDER_PROMPT,
        ref_mesh=config.RENDER_REF,
        distance=config.RENDER_RADIUS,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        min_delta=0.001,  # Minimum change to qualify as an improvement
        patience=config.PATIENCE,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,  # Whether to print logs to stdout
        mode='min'  # 'min' mode means training will stop when the quantity monitored stops decreasing
    )

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        logger=logger,
        accelerator='auto',
        callbacks=[
            #TQDMProgressBar(refresh_rate=20),
            early_stop_callback,
            render_callback
        ]
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
