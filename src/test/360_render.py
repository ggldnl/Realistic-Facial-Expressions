import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.renderer import Renderer


if __name__ == '__main__':

    """
    Renders a mesh. The mesh is supposed to be placed at the origin. We can 
    set the distance of the camera (radius) and the elevation. The viewpoints
    where the renders are computed are points equally spaced on the circumference
    around the mesh at the given elevation and radius. The number of points
    can be also set. The renders are stored in a directory as separate images.
    """

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Render a mesh from multiple viewpoints.")
    parser.add_argument('-m', '--mesh', type=str, required=True, help="Path to the mesh file (.obj).")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Directory to save rendered images.")
    parser.add_argument('-n', '--num_views', type=int, default=8, help="Number of views (default: 8).")
    parser.add_argument('-r', '--radius', type=int, default=600, help="Distance of the camera from the object (default: 600).")
    parser.add_argument('-e', '--elevation', type=int, default=0, help="Elevation angle (default: 0).")

    args = parser.parse_args()

    # Instantiate the renderer
    renderer = Renderer()

    # Mesh file path
    mesh_path = Path(args.mesh)

    # Output folder for the images
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full 360-degree scan, calculated based on num_views
    angle_step = 360 / args.num_views

    for i in range(args.num_views):

        angle = i * angle_step

        # Rt = Renderer.cylindrical_to_extrinsic(radius, elevation, angle)
        cam_pos, cam_view, up_vector = Renderer.cylindrical_to_pyvista(args.radius, args.elevation, angle)

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
