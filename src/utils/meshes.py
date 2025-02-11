from typing import Tuple

from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded

from pytorch3d.structures import Meshes
from pytorch3d.structures import list_to_padded
from pytorch3d.structures import list_to_packed
import torch


class WeightedMeshes(Meshes):
    """
    Extends PyTorch3D's Meshes class to add a per-vertex weights tensor.
    """

    def __init__(self, verts, faces, weights=None, **wkargs):
        """
        Initialize a WeightedMeshes object.

        Args:
            verts (torch.Tensor or list of torch.Tensor): 
                FloatTensor of shape (V, 3) for a single mesh 
                or a list of (Ni, 3) FloatTensors where V = sum(Ni).
            faces (torch.Tensor or list of torch.Tensor): 
                LongTensor of shape (F, 3) for a single mesh 
                or a list of (Mi, 3) LongTensors where F = sum(Mi).
            weights (torch.Tensor or list of torch.Tensor, optional): 
                FloatTensor of shape (V,) for a single mesh or a list of (Ni,) tensors. 
                Each weight corresponds to a vertex. Default is None, 
                in which case weights are initialized to 1.0 for all vertices.
        """
        super().__init__(verts, faces, **wkargs)

        if weights is None:
            self._weights_list = [torch.ones(v.shape[0], dtype=torch.float32, device=v.device) for v in self.verts_list()]
        else:
            if isinstance(weights, torch.Tensor):
                weights = [weights]
            if len(weights) != len(self.verts_list()):
                raise ValueError("Number of weight tensors must match the number of meshes.")
            for i, (v, w) in enumerate(zip(self.verts_list(), weights)):
                if v.shape[0] != w.shape[0]:
                    raise ValueError(
                        f"Mismatch in weights for mesh {i}: expected {v.shape[0]} weights, got {w.shape[0]}.")
            self._weights_list = weights

        self._weights_packed = None
        self._weights_padded = None

        self.num_meshes = len(verts)

    def weights_list(self):
        """Returns the weights as a list of tensors."""
        return self._weights_list

    def weights_packed(self):
        """Returns the packed tensor representation of weights."""
        self._compute_packed()
        return self._weights_packed

    def weights_padded(self):
        """Returns the padded tensor representation of weights."""
        self._compute_padded()
        return self._weights_padded

    def _compute_packed(self, refresh: bool = False):
        """Compute packed representation of weights."""
        super()._compute_packed(refresh=refresh)

        if not (refresh or self._weights_packed is None):
            return

        weights_list_to_packed = list_to_packed(self._weights_list)
        self._weights_packed = weights_list_to_packed[0].to(self.device)

    def _compute_padded(self, refresh: bool = False):
        """Compute padded representation of weights."""
        super()._compute_padded(refresh=refresh)

        if not (refresh or self._weights_padded is None):
            return

        self._weights_padded = list_to_padded(
            self._weights_list, (self._V, 1), pad_value=0.0, equisized=self.equisized
        )

    def clone(self):
        """
        Return a deep copy of the WeightedMeshes object.
        """
        new_mesh = super().clone()
        new_weights = [w.clone() for w in self._weights_list]
        return WeightedMeshes(new_mesh.verts_list(), new_mesh.faces_list(), new_weights)

    def sample_points(self, num_samples: int):
        """
        Sample points from the mesh surfaces with probability proportional to the face area,
        and return the sampled points along with their respective weights.

        Args:
            num_samples (int): The number of points to sample per mesh.

        Returns:
            tuple:
                - sampled_points (torch.Tensor): A tensor of shape (N, num_samples, 3)
                  containing the sampled points, where N is the number of meshes.
                - sampled_weights (torch.Tensor): A tensor of shape (N, num_samples)
                  containing the weights of the sampled points.
        """

        if self.isempty():
            raise ValueError("Meshes are empty.")

        verts = self.verts_packed()
        if not torch.isfinite(verts).all():
            raise ValueError("Meshes contain nan or inf.")

        faces = self.faces_packed()
        mesh_to_face = self.mesh_to_faces_packed_first_idx()
        num_meshes = self._N
        num_valid_meshes = torch.sum(self.valid)  # Non empty meshes.

        # Initialize samples tensor with fill value 0 for empty meshes.
        samples = torch.zeros((num_meshes, num_samples, 3), device=self.device)
        sampled_weights = torch.zeros((num_meshes, num_samples), device=self.device)

        # Only compute samples for non empty meshes
        with torch.no_grad():
            areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
            max_faces = self.num_faces_per_mesh().max().item()
            areas_padded = packed_to_padded(
                areas, mesh_to_face[self.valid], max_faces
            )  # (N, F)

            # TODO (gkioxari) Confirm multinomial bug is not present with real data.
            sample_face_idxs = areas_padded.multinomial(
                num_samples, replacement=True
            )  # (N, num_samples)
            sample_face_idxs += mesh_to_face[self.valid].view(num_valid_meshes, 1)

        # Get the vertex coordinates of the sampled faces.
        face_verts = verts[faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        # Randomly generate barycentric coords.
        w0, w1, w2 = WeightedMeshes._rand_barycentric_coords(
            num_valid_meshes, num_samples, verts.dtype, verts.device
        )

        # Use the barycentric coords to get a point on each sampled face.
        a = v0[sample_face_idxs]  # (N, num_samples, 3)
        b = v1[sample_face_idxs]
        c = v2[sample_face_idxs]
        samples[self.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

        # Compute the weights for the sampled points.
        weights = self.weights_packed()
        face_weights = weights[faces]
        w0_weights = face_weights[:, 0]  # Weights of vertex 0 for each face
        w1_weights = face_weights[:, 1]  # Weights of vertex 1 for each face
        w2_weights = face_weights[:, 2]  # Weights of vertex 2 for each face

        # Use barycentric coordinates to compute the sampled weights.
        wa = w0_weights[sample_face_idxs]  # (N, num_samples)
        wb = w1_weights[sample_face_idxs]
        wc = w2_weights[sample_face_idxs]
        sampled_weights[self.valid] = w0 * wa + w1 * wb + w2 * wc

        return samples, sampled_weights

    @staticmethod
    def _rand_barycentric_coords(
            size1, size2, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper function to generate random barycentric coordinates which are uniformly
        distributed over a triangle.

        Args:
            size1, size2: The number of coordinates generated will be size1*size2.
                          Output tensors will each be of shape (size1, size2).
            dtype: Datatype to generate.
            device: A torch.device object on which the outputs will be allocated.

        Returns:
            w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
                coordinates
        """
        uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
        u, v = uv[0], uv[1]
        u_sqrt = u.sqrt()
        w0 = 1.0 - u_sqrt
        w1 = u_sqrt * (1.0 - v)
        w2 = u_sqrt * v
        return w0, w1, w2


if __name__ == '__main__':

    from pathlib import Path

    from mesh_utils import read_mesh
    from mesh_utils import visualize_mesh

    root_dir = Path(__file__).parent.parent.parent
    mesh_path = Path(root_dir, 'datasets/facescape_highlighted/100/models_reg/1_neutral.obj')

    # Load the mesh
    mesh = read_mesh(mesh_path, normalize=True)
    print(f'Mesh loaded')

    # Sample the mesh
    sampled_points = mesh.sample_points(5000)
    print(f'Sampled points: {sampled_points}')

    visualize_mesh(mesh)
