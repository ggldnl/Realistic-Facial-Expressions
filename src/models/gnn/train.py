from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from src.data.text_generation import DEFAULT_TEXT_GENERATION
from src.data.datamodule import FacescapeDataModule
from src.models.gnn.model import Model
import src.models.gnn.config as config
from src.utils.callbacks import RenderCallback
from src.utils.renderer import Renderer

if __name__ == '__main__':

    torch.set_float32_matmul_precision('medium')

    datamodule = FacescapeDataModule(
        resource_url=config.RESOURCE_URL,
        download_source=config.DOWNLOAD_SOURCE,
        data_dir=config.DATA_DIR,
        download=config.DOWNLOAD,
        text_generation=DEFAULT_TEXT_GENERATION,
        batch_size=config.BATCH_SIZE,
        first_subject=config.FIRST_SUBJECT_INDEX,
        last_subject=config.LAST_SUBJECT_INDEX
    )


    logger = TensorBoardLogger(config.LOG_DIR, name="GNN")


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
        monitor='val_loss',
        min_delta=0.001,
        patience=config.PATIENCE,
        verbose=True,
        mode='min'
    )

    # Add ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Metric to monitor
        dirpath=config.CHECKPOINT_DIR,  # Directory to save checkpoints
        filename='gnn-{epoch:02d}-{val_loss:.4f}',  # Filename format
        save_top_k=3,  # Save the top 3 models with the lowest val_loss
        mode='min'  # Minimize val_loss
    )

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        logger=logger,
        accelerator='auto',
        callbacks=[
            early_stop_callback,
            render_callback,
            checkpoint_callback
        ]
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
