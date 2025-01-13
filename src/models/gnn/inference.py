import argparse

from src.utils.mesh_utils import visualize_mesh
from src.utils.mesh_utils import read_meshes
from src.models.gnn.model import Model
from src.models.gnn import config
import torch

if __name__ == '__main__':

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Perform inference on the given mesh + text description.")
    parser.add_argument('-m', '--mesh', type=str, required=True, help="Path to the input neutral mesh.")
    parser.add_argument('-t', '--text', type=str, required=True, help="Text description of the expression we want.")

    args = parser.parse_args()

    input_meshes = read_meshes([args.mesh])
    descriptions = [args.text]

    """
    # Create the model
    model = Model(
        latent_size=config.LATENT_SIZE,
        input_dim=config.INPUT_DIM,
        batch_size=1
    )
    """

    # Restore the model
    checkpoint_path = config.CHECKPOINT_DIR / "gnn-epoch=19-val_loss=0.0298.ckpt"
    model = Model.load_from_checkpoint(checkpoint_path)

    model.eval()
    # Perform inference
    output_mesh = model.inference(input_meshes.to("cuda" if torch.cuda.is_available() else "cpu"), descriptions)

    # Visualize the result
    visualize_mesh(output_mesh)
