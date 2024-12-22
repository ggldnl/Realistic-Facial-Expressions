import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.renderer import Renderer
import src.config.config as config


if __name__ == "__main__":

    # Instantiate the renderer
    renderer = Renderer()

    data_dir = Path(__file__).parent.parent.parent / "datasets" / "facescape"

    # Render the mesh using PyVista
    mesh_path = Path(data_dir, '100/models_reg/1_neutral.obj')

    # Output folder for the images
    output_dir = Path(data_dir, "100/models_reg/1_neutral_renders")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full 360-degree scan, every 45 degrees
    num_views = 8  # 45 degrees
    angle_step = 360 / num_views
    radius = 600  # Distance of the camera from the object
    elevation = 0

    for i in range(num_views):

        angle = i * angle_step

        # Rt = Renderer.cylindrical_to_extrinsic(radius, elevation, angle)
        cam_pos, cam_view, up_vector = Renderer.cylindrical_to_pyvista(radius, elevation, angle)

        # Render the image
        rendered_img = renderer.render(
            model_in=mesh_path,
            cam_pos=cam_pos,
            cam_view=cam_view,
            up_vector=up_vector,
            scale=1.0,
            rend_size=(1024, 768)
        )

        # Save the image
        output_path = output_dir / f"view_{i:02d}.png"
        plt.imsave(output_path, rendered_img)
        print(f"Saved view {i} at {output_path}")
