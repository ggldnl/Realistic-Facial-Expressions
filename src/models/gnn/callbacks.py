from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


from src.utils.renderer import Renderer
from src.utils.mesh_utils import read_graph
from src.utils.mesh_utils import tensor_to_mesh


class RenderCallback(Callback):
    """
    Callback that is called once every n epochs that renders a test mesh
    from the standard viewpoints (each 45 degrees) and saves the result
    in the specified folder, just to check the training is working.
    """

    def __init__(self,
                 n_epochs,  # Number of epochs after which we perform the rendering
                 model,  # Necessary to perform inference
                 in_mesh,  # (Neutral) mesh to start with
                 out_dir,  # Folder where to save the results
                 prompt='smile',  # Prompt used to modify the neutral mesh during inference

                 # Rendering options
                 n_viewpoints=8,  # Number of renders to take
                 distance=600,  # Distance from the center (= distance from the mesh)
                 elevation=0,  # Elevation (used during rendering)
                 render_w=600,
                 render_h=800,
                 ):
        self.n_epochs = n_epochs
        self.model = model
        self.prompt = prompt
        self.out_dir = out_dir

        self.in_mesh = in_mesh if isinstance(in_mesh, Path) else Path(in_mesh)
        assert "neutral" in self.in_mesh.stem, "The provided file appears not to be a valid neutral mesh."

        self.n_viewpoints = n_viewpoints
        self.distance = distance
        self.elevation = elevation
        self.render_w = render_w
        self.render_h = render_h

        self.renderer = Renderer()


    def on_validation_epoch_end(self, trainer, pl_module):

        epoch = trainer.current_epoch
        if epoch % self.n_epochs == 0:

            print(f'\nPerforming inference on neutral mesh {self.in_mesh} with prompt: \'{self.prompt}\'')

            neutral_mesh = read_graph(self.in_mesh)
            x = self.model(neutral_mesh, [self.prompt])
            pred_mesh = tensor_to_mesh(x.squeeze(0), neutral_mesh.edge_index)  # Batch containing a single mesh -> squeeze batch
            print(f'Inference completed. Performing rendering...')

            # Full 360-degree scan, every 45 degrees
            angle_step = 360 / self.n_viewpoints

            renders = []
            for i in range(self.n_viewpoints):

                angle = i * angle_step

                cam_pos, cam_view, up_vector = Renderer.cylindrical_to_pyvista(self.distance, self.elevation, angle)

                # Render the image
                rendered_img = self.renderer.render(
                    model_in=pred_mesh,
                    cam_pos=cam_pos,
                    cam_view=cam_view,
                    up_vector=up_vector,
                    scale=1.0,
                    rend_size=(self.render_w, self.render_h)
                )

                renders.append(rendered_img)

            print(f'Rendering completed. Saving the result...')

            # Stitch together the images
            stitched_image = np.concatenate(renders, axis=1)

            # Display the stitched image
            plt.figure(figsize=(12, 6))
            plt.imshow(stitched_image, cmap='gray')  # Use 'gray' if grayscale images
            plt.axis('off')  # Hide axes

            # Save the image
            output_path = Path(self.out_dir, f"epoch_{epoch:02d}.png")
            plt.imsave(output_path, stitched_image)
            print(f"Saved renders for epoch {epoch} at {output_path}.")
