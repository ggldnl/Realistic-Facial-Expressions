import numpy as np
import pyvista

from src.utils.decorators import fixme


class Renderer:

    def __init__(self):
        pass

    @staticmethod
    def _load_mesh(model_in):
        return trimesh.load(model_in, process=False)

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

    def render(self,
               model_in,
               Rt=None,
               cam_pos=None,
               cam_view=None,
               up_vector=None,
               scale=1.0,
               rend_size=(512, 512)
    ):
        """
        Renders the mesh using PyVista. PyVista uses a camera model where the camera position, direction,
        and up vector are specified directly, and it handles the intrinsic details internally, so the
        intrinsic camera matrix is not used. You can specify the Rt matrix (intrinsic parameters) for the
        camera position and direction
        """

        if Rt is not None and (cam_pos is not None or cam_view is not None or up_vector is not None):
            raise ValueError('Multiple ways to compute camera position and orientation provided.')

        # Load the mesh using trimesh
        mesh = self._load_mesh(model_in)

        # Each face must start with the number of vertices (3 for triangles)
        faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces])  # Prefix faces with 3 (triangle)
        faces = faces.astype(np.int64)

        # Convert to PyVista mesh
        pv_mesh = pyvista.PolyData(mesh.vertices, faces)

        # Scale the mesh
        pv_mesh.scale([scale, scale, scale], inplace=True)

        # Create a PyVista plotter in offscreen mode
        plotter = pyvista.Plotter(window_size=rend_size, off_screen=True)

        # Configure camera using Rt if provided
        if Rt is not None:
            """
            cam_pos = Rt[:3, 3]  # Camera position
            cam_view = [0, 0, 0]  # Rt[:3, 3] - Rt[:3, 2]  # Look-at point (camera position minus z-axis)
            up_vector = Rt[:3, 1]  # Up vector
            """
            cam_pos, cam_view, up_vector = self.extrinsic_to_pyvista(Rt)

        if cam_pos is not None and cam_view is not None and up_vector is not None:
            plotter.camera_position = [cam_pos, cam_view, up_vector]

        # Add the mesh to the plotter
        plotter.clear()
        plotter.add_mesh(pv_mesh, color="lightblue", show_edges=True)

        # Render the scene and capture the image as a numpy array
        img = plotter.screenshot(return_img=True)

        # Close the plotter to release resources
        plotter.close()

        # Return the rendered image
        return img


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from pathlib import Path
    import trimesh

    import src.config.config as config

    # Instantiate the renderer
    renderer = Renderer()

    # Render using the OpenGL camera method
    mesh_path = Path(config.DATA_DIR, '100/models_reg/1_neutral.obj')

    # Render the mesh using PyVista and get the image
    rendered_img = renderer.render(
        model_in=mesh_path,
        scale=1.0,
        rend_size=(1024, 768)
    )

    # Display the rendered image using matplotlib
    plt.imshow(rendered_img)
    plt.title("Rendered Mesh")
    plt.axis("off")
    plt.show()
