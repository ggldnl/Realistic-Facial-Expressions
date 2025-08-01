from pathlib import Path
import numpy as np
import pyvista
import trimesh
from pytorch3d.structures import Meshes
from trimesh import Trimesh

from src.utils.decorators import fixme


class Renderer:

    def __init__(self):
        pass

    @staticmethod
    def extrinsic_to_pyvista(extrinsic_matrix):
        """
        Converts a standard 4x4 transformation matrix into PyVista camera parameters
        """
        if extrinsic_matrix.shape != (4, 4):
            raise ValueError("Extrinsic matrix must be a 4x4 numpy array.")

        # Extract the rotation matrix (3x3) and the translation vector (3x1)
        R = extrinsic_matrix[:3, :3]  # Rotation matrix
        t = extrinsic_matrix[:3, 3]  # Translation vector

        # Camera position (translation vector, in world coordinates)
        camera_position = -np.linalg.inv(R) @ t

        # Focal point (assuming the camera looks towards the origin in world coordinates)
        focal_point = [0, 0, 0]

        # Up vector (third column of the rotation matrix in world coordinates)
        up_vector = R[:, 1]

        return camera_position, focal_point, up_vector

    @staticmethod
    def cylindrical_to_pyvista(radius, elevation, angle_deg):
        """
        Create the pyvista parameter that describes the camera at a given distance(radius), elevation and
        angle around the z axis
        """

        angle_rad = np.radians(angle_deg)

        # Compute camera position
        cam_x = radius * np.cos(angle_rad)
        cam_z = radius * np.sin(angle_rad)
        cam_y = elevation
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # Compute forward vector (camera looks at the origin)
        forward = -cam_pos / np.linalg.norm(cam_pos)

        # Define up vector
        up = np.array([0, 1, 0])  # Y-axis is up
        right = np.cross(up, forward)  # Right vector
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)  # Recompute up for orthogonality

        return cam_pos, forward, up

    @staticmethod
    @fixme
    def cylindrical_to_extrinsic(radius, elevation, angle_deg):
        """
        Converts from cylindrical coordinates to a standard 4x4 transformation matrix
        """

        # Convert angle to radians
        angle_rad = np.radians(angle_deg)

        # Compute rotation matrix around Z-axis
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Compute translation vector
        translation_vector = np.array([
            [1, 0, 0, radius * np.cos(angle_rad)],
            [0, 1, 0, radius * np.sin(angle_rad)],
            [0, 0, 1, elevation],
            [0, 0, 0, 1]
        ])

        # Combine rotation and translation
        transformation_matrix = translation_vector @ rotation_matrix

        return transformation_matrix

    def render_viewpoints(self,
                          model_in,
                          num_views=8,
                          radius=600,
                          elevation=0,
                          scale=1.0,
                          rend_size=(1024, 768),
                          return_images=False
                          ):
        """
        Renders multiple views of the mesh by rotating the camera around the object.

        Args:
            model_in: Path to the input 3D model file
            num_views (int): Number of views to render (default: 8)
            radius (float): Distance of camera from the object (default: 600)
            elevation (float): Height of camera above the object (default: 0)
            scale (float): Scale factor for the mesh (default: 1.0)
            rend_size (tuple): Resolution of rendered images (width, height) (default: (1024, 768))
            return_images (bool): If True, return raw images else return pyvista objects

        Returns:
            list: List of numpy arrays containing the rendered images
        """
        rendered_images = []
        angle_step = 360 / num_views

        for i in range(num_views):
            angle = i * angle_step

            cam_pos, cam_view, up_vector = self.cylindrical_to_pyvista(radius, elevation, angle)

            rendered_img = self.render(
                model_in=model_in,
                cam_pos=cam_pos,
                cam_view=cam_view,
                up_vector=up_vector,
                scale=scale,
                rend_size=rend_size,
                return_as_image=return_images
            )

            rendered_images.append(rendered_img)

        return rendered_images

    def render(self,
               model_in,
               Rt=None,
               cam_pos=None,
               cam_view=None,
               up_vector=None,
               scale=1.0,
               rend_size=(512, 512),
               return_as_image=False,
               show_edges=True,
               foreground_color="lightblue",
               background_color="white",
        ):
        """
        Renders the mesh using PyVista with options for foreground and background colors,
        and optionally with a transparent background.
        """

        if Rt is not None and (cam_pos is not None or cam_view is not None or up_vector is not None):
            raise ValueError('Multiple ways to compute camera position and orientation provided.')

        if isinstance(model_in, str) or isinstance(model_in, Path):
            # Load the mesh using trimesh
            mesh = trimesh.load(model_in, process=False)
        else:
            mesh = model_in

        if isinstance(mesh, Trimesh):
            faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces])  # Prefix faces with 3 (triangle)
            faces = faces.astype(np.int64)
            vertices = mesh.vertices
        elif isinstance(mesh, Meshes):
            faces = mesh.faces_list()[0].cpu().numpy().astype(np.int64)
            faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])  # Add a prefix with the number of nodes
            vertices = mesh.verts_list()[0].cpu().numpy()
        else:
            raise ValueError(f'Unsupported type: {type(mesh)}')

        # Convert to PyVista mesh
        pv_mesh = pyvista.PolyData(vertices, faces)

        # Scale the mesh
        pv_mesh.scale([scale, scale, scale], inplace=True)

        # Create a PyVista plotter in offscreen mode
        plotter = pyvista.Plotter(window_size=rend_size, off_screen=True)

        # Configure camera using Rt if provided
        if Rt is not None:
            cam_pos, cam_view, up_vector = self.extrinsic_to_pyvista(Rt)

        if cam_pos is not None and cam_view is not None and up_vector is not None:
            plotter.camera_position = [cam_pos, cam_view, up_vector]

        # Set the background color
        transparent_background = False
        if background_color == 'transparent':
            transparent_background = True
            background_color = 'white'  # PyVista does not natively support transparent backgrounds

        plotter.set_background(background_color)

        # Add the mesh to the plotter
        plotter.clear()
        plotter.add_mesh(pv_mesh, color=foreground_color, show_edges=show_edges)

        # Render the scene and capture the image as a numpy array
        img = plotter.screenshot(return_img=True)

        # Close the plotter to release resources
        plotter.close()

        # Handle transparency
        if transparent_background:
            from PIL import Image

            # Convert the image to RGBA and replace the background color with transparency
            img = Image.fromarray(img).convert("RGBA")
            data = np.array(img)
            red, green, blue, alpha = data.T
            # Replace background color (assumed white here) with transparency
            white_areas = (red == 255) & (green == 255) & (blue == 255)
            data[..., :-1][white_areas.T] = (0, 0, 0)  # Black for fully transparent areas
            data[..., -1][white_areas.T] = 0  # Set alpha to 0
            img = Image.fromarray(data)

            if return_as_image:
                return img

        if return_as_image:
            from PIL import Image
            return Image.fromarray(img)

        return img


if __name__ == "__main__":

    from pathlib import Path
    import matplotlib.pyplot as plt

    root_dir = Path(__file__).parent.parent.parent
    data_dir = Path(root_dir, 'datasets/facescape')

    # Instantiate the renderer
    renderer = Renderer()

    # Render using the OpenGL camera method
    mesh_path = Path(data_dir, '100/models_reg/1_neutral.obj')

    # Render the mesh using PyVista and get the image
    rendered_img = renderer.render(
        model_in=mesh_path,
        scale=1.0,
        rend_size=(1024, 768),
        foreground_color='red',
        background_color='transparent',
        show_edges=True
    )

    # Display the rendered image using matplotlib
    plt.imshow(rendered_img)
    plt.title("Rendered Mesh")
    plt.axis("off")
    plt.show()
