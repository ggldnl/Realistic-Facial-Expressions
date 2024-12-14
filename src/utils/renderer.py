import numpy as np
import pyvista

from src.utils.decorators import deprecated


class Renderer:

    def __init__(self):
        pass

    @staticmethod
    def _load_mesh(model_in):
        return trimesh.load(model_in, process=False)

    def render_pyvista(self, model_in, K=None, Rt=None, scale=1.0, rend_size=(512, 512)):
        """
        Renders the mesh using PyVista. PyVista uses a different camera model where the camera position,
        view direction, and up vector are specified directly, and it handles the intrinsic details internally,
        so the intrinsic camera matrix is not used.
        """

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
        plotter.add_mesh(pv_mesh, color="lightblue", show_edges=True)

        # Configure camera using Rt if provided
        if Rt is not None:
            cam_pos = Rt[:3, 3]  # Camera position
            cam_view = cam_pos - Rt[:3, 2]  # Look-at point
            up_vector = Rt[:3, 1]  # Up vector
            plotter.camera_position = [cam_pos, cam_view, up_vector]

        # Render the scene and capture the image as a numpy array
        img = plotter.screenshot(return_img=True)

        # Close the plotter to release resources
        plotter.close()

        # Return the rendered image
        return img

    @deprecated
    def render_glcam(self, model_in, K, Rt, scale=1.0, rend_size=(512, 512),
                     light_trans=np.array([[0], [100], [0]]), flat_shading=False):
        """
        Renders the mesh using OpenGL's coordinate system conventions: Z-axis is forward and the Y-axis is up.
        """

        # Mesh creation
        mesh = self._load_mesh(model_in)
        pr_mesh = pyrender.Mesh.from_trimesh(mesh)

        # Scene creation
        scene = pyrender.Scene()
        scene.add(pr_mesh)

        # Calculate camera intrinsics
        fx, fy = K[0][0] * scale, K[1][1] * scale
        cx, cy = K[0][2] * scale, K[1][2] * scale

        # Camera setup
        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.1, zfar=100000)
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = Rt[:3, :3].T
        cam_pose[:3, 3] = -Rt[:3, :3].T.dot(Rt[:, 3])
        scene.add(cam, pose=cam_pose)

        # Light setup
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        light_pose = cam_pose.copy()
        light_pose[0:3, :] += light_trans
        scene.add(light, pose=light_pose)

        # Rendering
        renderer = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                              viewport_height=rend_size[0],
                                              point_size=1.0)
        flags = pyrender.constants.RenderFlags.FLAT if flat_shading else 0
        color, depth = renderer.render(scene, flags=flags)

        # Convert RGB to BGR
        color = color[:, :, [2, 1, 0]]

        return depth, color

    @deprecated
    def render_cvcam(self, model_in, K=None, Rt=None, scale=1.0, rend_size=(512, 512),
                      light_trans=np.array([[0], [100], [0]]), flat_shading=False):
        """
        OpenCV's format.
        """

        if K is None:
            K = np.array([[2000, 0, 256],
                    [0, 2000, 256],
                    [0, 0, 1]], dtype=np.float64)
        if Rt is None:
            Rt = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]], dtype=np.float64)

        # Transform Rt for OpenCV to OpenGL
        R_cv2gl = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]])
        Rt_cv = R_cv2gl.dot(Rt)

        return self.render_glcam(model_in, K, Rt_cv, scale, rend_size, light_trans, flat_shading)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from pathlib import Path
    import trimesh
    import json

    import src.config.config as config

    # Instantiate the renderer
    renderer = Renderer()

    # Read the params file (given in the dataset) and take a config dict
    params_path = Path(config.DATA_DIR, "100/1_neutral/params.json")
    with open(params_path) as f:
        params = json.load(f)

    # There is a parameter dict in the same file for each image of the rig (60 in total, from 0 to 59).
    # This for each expression of each user.
    id = 0
    K = np.array(params[f'{id}_K'])
    Rt = np.array(params[f'{id}_Rt'])

    # Render using the OpenGL camera method
    mesh_path = Path(config.DATA_DIR, '100/models_reg/1_neutral.obj')

    # Render the mesh using PyVista and get the image
    rendered_img = renderer.render_pyvista(
        model_in=mesh_path,
        scale=1.0,
        rend_size=(1024, 768)
    )

    # Display the rendered image using matplotlib
    plt.imshow(rendered_img)
    plt.title("Rendered Mesh")
    plt.axis("off")
    plt.show()
