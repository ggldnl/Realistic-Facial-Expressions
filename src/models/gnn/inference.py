import argparse

from src.utils.mesh_utils import visualize_mesh
from src.utils.mesh_utils import read_meshes
from src.models.gnn.model import Model
from src.models.gnn import config


if __name__ == '__main__':

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Perform inference on the given mesh + text description.")
    parser.add_argument('-m', '--mesh', type=str, required=True, help="Path to the input neutral mesh.")
    parser.add_argument('-t', '--text', type=str, required=True, help="Text description of the expression we want.")
    parser.add_argument('-o', '--output', type=str, default=None, help="Path to the file where to save the resulting mesh.")

    args = parser.parse_args()

    input_meshes = read_meshes([args.mesh])
    descriptions = [args.text]
    output_path = args.output

    # Create the model
    model = Model(
        latent_size=config.LATENT_SIZE,
        input_dim=config.INPUT_DIM,
        batch_size=1
    )

    # Restore the model
    # TODO
    model = Model.load_from_checkpoint(...)

    # Perform inference
    output_mesh = model.inference(input_meshes, descriptions)

    # Visualize the result
    visualize_mesh(output_mesh)

    if output_path is not None:
        print('eheh')
