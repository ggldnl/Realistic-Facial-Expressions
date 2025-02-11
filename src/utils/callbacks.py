from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
from src.utils.mesh_utils import read_meshes


class RenderCallback(Callback):
    """
    Callback that is called once every n epochs that renders a test mesh
    from the standard viewpoints (each 45 degrees) and saves the result
    in the specified folder, just to check the training is working.
    """

    def __init__(self,
                 n_epochs,  # Number of epochs after which we perform the rendering
                 model,  # Necessary to perform inference
                 renderer,  # Mesh renderer
                 in_mesh,  # (Neutral) mesh to start with
                 out_dir,  # Folder where to save the results
                 ref_mesh=None,  # Optional mesh to render below the predicted
                 prompt='smile',  # Prompt used to modify the neutral mesh during inference

                 # Rendering options
                 n_viewpoints=8,  # Number of renders to take
                 distance=5,  # Distance from the center (= distance from the mesh)
                 elevation=0,  # Elevation (used during rendering)
                 render_w=600,
                 render_h=800,
                 ):
        super().__init__()
        self.n_epochs = n_epochs
        self.model = model
        self.renderer = renderer
        self.prompt = prompt
        self.out_dir = out_dir

        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)

        self.out_dir.mkdir(exist_ok=True, parents=True)

        self.in_mesh = in_mesh if isinstance(in_mesh, Path) else Path(in_mesh)
        assert "neutral" in self.in_mesh.stem, "The provided file appears not to be a valid neutral mesh."

        self.ref_mesh = ref_mesh if isinstance(in_mesh, Path) else Path(in_mesh)

        self.n_viewpoints = n_viewpoints
        self.distance = distance
        self.elevation = elevation
        self.render_w = render_w
        self.render_h = render_h

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.n_epochs == 0:
            print(f'\nPerforming inference on neutral mesh {self.in_mesh} with prompt: \'{self.prompt}\'')

            self.model.eval()

            meshes = read_meshes([self.in_mesh], normalize=True)
            meshes = meshes.to(pl_module.device)

            # Use torch.no_grad() to disable gradient computation
            with torch.no_grad():
                displacements = self.model(meshes, [self.prompt])
                pred_mesh = meshes.offset_verts(displacements)

            print('Inference completed. Performing rendering...')

            renders = self.renderer.render_viewpoints(model_in=pred_mesh,
                                                      num_views=self.n_viewpoints,
                                                      radius=self.distance,
                                                      elevation=self.elevation,
                                                      scale=1.0,
                                                      rend_size=(self.render_w, self.render_h))

            expression_renders = None
            neutral_renders = None
            if self.ref_mesh:
                # Read the expression mesh if provided
                expression_meshes = read_meshes([self.ref_mesh], normalize=True)

                # Render the expression mesh
                expression_renders = self.renderer.render_viewpoints(model_in=expression_meshes,
                                                                     num_views=self.n_viewpoints,
                                                                     radius=self.distance,
                                                                     elevation=self.elevation,
                                                                     scale=1.0,
                                                                     rend_size=(self.render_w, self.render_h))

                # Render the neutral mesh
                neutral_renders = self.renderer.render_viewpoints(model_in=meshes,
                                                                  num_views=self.n_viewpoints,
                                                                  radius=self.distance,
                                                                  elevation=self.elevation,
                                                                  scale=1.0,
                                                                  rend_size=(self.render_w, self.render_h))

            print('Rendering completed. Saving the result...')

            # Stitch together the images
            morphed_image = np.concatenate(renders, axis=1)

            if expression_renders and neutral_renders:
                stitched_expression_image = np.concatenate(expression_renders, axis=1)
                stitched_neutral_image = np.concatenate(neutral_renders, axis=1)
                morphed_image = np.concatenate([
                    stitched_neutral_image,
                    morphed_image,
                    stitched_expression_image
                ], axis=0)

            # Save the image
            output_path = Path(self.out_dir, f"epoch_{epoch:02d}.png")
            plt.imsave(output_path, morphed_image)
            print(f"Saved renders for epoch {epoch} at {output_path}.")
