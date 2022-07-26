from models.frameworks.neumesh import MeshSampler
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d


def get_bbox_on_uv(maskmesh_uv):
    valid_uv = maskmesh_uv
    domain_min = [
        valid_uv[..., 0].min(),
        valid_uv[..., 1].min(),
    ]
    domain_max = [
        valid_uv[..., 0].max(),
        valid_uv[..., 1].max(),
    ]
    return np.array(domain_min), np.array(domain_max)


class EditablePrimitive:
    def __init__(
        self,
        model,
        masks,
        maskmeshs_uv,
        maskmeshs_uv2vertices,
        geo_feature=None,
        color_feature=None,
    ):
        self.model = model
        self.masks = masks
        self.maskmeshs_uv = maskmeshs_uv
        self.maskmeshs_uv2vertices = maskmeshs_uv2vertices
        if color_feature is not None:
            self.edit_color_features = torch.zeros_like(color_feature)
        else:
            self.edit_color_features = None

        if geo_feature is not None:
            self.edit_geo_features = torch.zeros_like(geo_feature)
        else:
            self.edit_geo_features = None

        if maskmeshs_uv is not None:
            assert len(self.masks) == len(self.maskmeshs_uv)

        if maskmeshs_uv2vertices is not None:
            assert len(self.masks) == len(self.maskmeshs_uv2vertices)

    def get_masked_mesh(self, i=0):
        main_mesh_trans = o3d.geometry.TriangleMesh()
        main_mesh = self.get_mesh()
        main_mesh_trans.vertices = main_mesh.vertices
        main_mesh_trans.triangles = main_mesh.triangles
        main_color = np.array(main_mesh.vertex_colors)
        main_color[self.get_mask(i) == False, :] = 0
        main_mesh_trans.vertex_colors = o3d.utility.Vector3dVector(main_color)
        return main_mesh_trans

    def compute_distance_in_masked_region(self, xyz, i=0):
        INF = 10000
        masked_mesh = o3d.geometry.TriangleMesh()
        mesh = self.get_mesh()
        vertices = np.array(mesh.vertices)
        mask = self.get_mask(i)
        number2vertexindices = np.where(mask)[0]
        vertexindices2number = [
            (number2vertexindices[i], i) for i in range(len(number2vertexindices))
        ]
        vertexindices2number = dict(vertexindices2number)
        masked_mesh.vertices = o3d.utility.Vector3dVector(
            vertices[number2vertexindices]
        )
        triangles = np.array(mesh.triangles)
        mask_triangles = np.all(mask[triangles], axis=-1)
        triangles = triangles[mask_triangles].flatten()
        for i in range(len(triangles)):
            triangles[i] = vertexindices2number[triangles[i]]
        triangles = triangles.reshape(-1, 3)
        masked_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        masked_mesh_sampler = MeshSampler(
            masked_mesh,
            device="cuda",
            distance_method=self.model.mesh_sampler.distance_method,
            minimum_box=self.model.mesh_sampler.minimum_box,
        )
        _ds, _indices, _weights = masked_mesh_sampler.compute_distance(xyz)
        return (
            _ds,
            torch.from_numpy(number2vertexindices[_indices.detach().cpu().numpy()]).to(
                _weights.device
            ),
            _weights,
        )

    def get_mask(self, i=0):
        return self.masks[i]

    def get_maskmesh_uv(self, i=0):
        return self.maskmeshs_uv[i]

    def get_maskmesh_uv2vertices(self, i=0):
        return self.maskmeshs_uv2vertices[i]

    def get_len_of_mask(self):
        return len(self.masks)

    def to_tensor(self, device=None):
        for i in range(len(self.masks)):
            self.mask_to_tensor(i)

        if (
            self.edit_geo_features is not None
            and torch.is_tensor(self.edit_geo_features) == False
        ):
            self.edit_geo_features = torch.from_numpy(self.edit_geo_features)
        if (
            self.edit_color_features is not None
            and torch.is_tensor(self.edit_color_features) == False
        ):
            self.edit_color_features = torch.from_numpy(self.edit_color_features)

        if device is not None:
            self.to_cuda(device)

    def to_cuda(self, device):
        for i in range(len(self.masks)):
            self.mask_to_cuda(i, device)
        self.model = self.model.to(device)
        if self.edit_geo_features is not None:
            self.edit_geo_features = self.edit_geo_features.to(device)
        if self.edit_color_features is not None:
            self.edit_color_features = self.edit_color_features.to(device)

    def mask_to_tensor(self, i=0):
        self.masks[i] = torch.from_numpy(self.masks[i])
        if self.maskmeshs_uv is not None:
            self.maskmeshs_uv[i] = torch.from_numpy(self.maskmeshs_uv[i])
        if self.maskmeshs_uv2vertices is not None:
            self.maskmeshs_uv2vertices[i] = torch.from_numpy(
                self.maskmeshs_uv2vertices[i]
            )

    def mask_to_cuda(self, i, device):
        self.masks[i] = self.masks[i].to(device)
        if self.maskmeshs_uv is not None:
            self.maskmeshs_uv[i] = self.maskmeshs_uv[i].to(device)
        if self.maskmeshs_uv2vertices is not None:
            self.maskmeshs_uv2vertices[i] = self.maskmeshs_uv2vertices[i].to(device)

    def get_bbox_on_uv(self, i=0):
        return get_bbox_on_uv(self.get_maskmesh_uv(i))

    def get_mesh_sampler(self):
        return self.model.mesh_sampler

    def get_mesh(self):
        return self.model.mesh_sampler.mesh

    def get_mesh_vertices(self, b_torch=True):
        if b_torch == False:
            return np.array(self.get_mesh().vertices)
        else:
            return self.model.mesh_sampler.mesh_vertices

    def crop_uv(self, i, domain_min, domain_max):
        maskmesh_uv = self.maskmeshs_uv[i]
        is_inside = (
            (maskmesh_uv[..., 0] > domain_min[0])
            & (maskmesh_uv[..., 0] < domain_max[0])
            & (maskmesh_uv[..., 1] > domain_min[1])
            & (maskmesh_uv[..., 1] < domain_max[1])
        )
        self.maskmeshs_uv[i] = maskmesh_uv[is_inside]
        self.maskmeshs_uv2vertices[i] = self.maskmeshs_uv2vertices[i][is_inside]
        self.masks[i] = self.masks[i] & False
        self.masks[i][self.maskmeshs_uv2vertices[i]] = True

    def get_uvcolor(self, i, bgr=False):
        rgb_colors = (
            np.array(self.get_mesh().vertex_colors)[self.maskmeshs_uv2vertices[i]] * 255
        )
        if bgr == True:
            rgb_colors[..., [0, 2]] = rgb_colors[..., [2, 0]]
        return rgb_colors


def transform_vertices(rot_s_m, t_s_m, xyz):
    rot_s_m_brc = rot_s_m[(slice(None),) * 0 + (None,) * (xyz.dim() - 1)]
    t_s_m_brc = t_s_m[(slice(None),) * 0 + (None,) * (xyz.dim() - 1)]
    slave_xyz = torch.matmul(rot_s_m_brc, xyz.unsqueeze(-1)).squeeze(-1) + t_s_m_brc
    return slave_xyz


def transform_direction(rot_s_m, dirs):
    rot_s_m_brc = rot_s_m[(slice(None),) * 0 + (None,) * (dirs.dim() - 1)]
    slave_dir = torch.matmul(rot_s_m_brc, dirs.unsqueeze(-1)).squeeze(-1)
    return slave_dir


class TextureEditableNeuMesh(nn.Module):
    def __init__(
        self,
        main_primitive,
        slave_primitives,
        T_s_m_list=None,
        method="code",
        blend_black=False,
        blur_edge=False,
    ):
        super(TextureEditableNeuMesh, self).__init__()
        self.main_primitive = main_primitive
        self.slave_primitives = slave_primitives
        if T_s_m_list is not None:
            self.rot_s_m = []
            self.t_s_m = []
            for T_s_m in T_s_m_list:
                self.rot_s_m.append(T_s_m[:3, :3])
                self.t_s_m.append(T_s_m[:3, 3])
        else:
            self.rot_s_m = None
            self.t_s_m = None

        self.method = method
        self.blend_black = blend_black
        self.enable_nablas_input = main_primitive.model.enable_nablas_input
        self.blur_edge = blur_edge

    def compute_distance(self, xyz):
        return self.main_primitive.model.compute_distance(xyz)

    def forward_s(self):
        return self.main_primitive.model.forward_s()

    def forward_density_only(self, xyz, view_dirs):
        return self.main_primitive.model.forward_density_only(xyz, view_dirs)

    def compute_bounded_near_far(
        self,
        rays_o,
        rays_d,
        near,
        far,
        sample_grid: int = 256,
        distance_thresh: float = 0.1,
    ):
        return self.main_primitive.model.compute_bounded_near_far(
            rays_o, rays_d, near, far, sample_grid, distance_thresh
        )

    def forward_with_nablas(self, xyz: torch.Tensor, view_dirs: torch.Tensor):
        return self.main_primitive.model.forward(
            xyz,
            view_dirs,
            density_only=False,
            need_nablas=True,
            nablas_only=True,
        )

    def forward(
        self,
        xyz: torch.Tensor,
        view_dirs: torch.Tensor,
        density_only=False,
        need_nablas=True,
        nablas_only=False,
    ):
        """
        xyz: (...,3)
        dirs: (...,3)
        """
        (
            sdf,
            nabla,
            _ds,
            _indices,
            _weights,
        ) = self.main_primitive.model.forward_with_ds(xyz, view_dirs, False, True, True)
        colors = self.main_primitive.model.forward_editcolor(
            self.main_primitive.model.color_features,
            _ds,
            view_dirs,
            indices=_indices,
            weights=_weights,
            nabla=nabla,
        )

        blend_color = colors.clone()
        _raw_ds = torch.linalg.norm(
            xyz.unsqueeze(-2)
            - self.main_primitive.model.mesh_sampler.mesh_vertices[_indices],
            dim=-1,
        )  # (..., K)
        for i in range(len(self.slave_primitives)):
            slave_primitive = self.slave_primitives[i]
            main_mask = self.main_primitive.get_mask(i)
            # b. compute blending weights
            if self.blur_edge == True:
                paint_weight = torch.sum(_weights * main_mask[_indices], dim=-1)
                unpaint_weight = torch.sum(
                    _weights * (main_mask[_indices] == False), dim=-1
                )
                paint_region = paint_weight > 0
                sum_weight = paint_weight + unpaint_weight
                paint_weight /= sum_weight
                unpaint_weight /= sum_weight
                paint_weight = paint_weight[paint_region]
                unpaint_weight = unpaint_weight[paint_region]
            else:
                paint_region = torch.sum(main_mask[_indices], dim=-1) >= 8

            slave_weights = _weights * main_mask[_indices]
            slave_weights /= torch.sum(slave_weights, dim=-1, keepdim=True) + 1e-8

            # c. query slave color
            if self.rot_s_m is not None and self.t_s_m is not None:
                rot_s_m = self.rot_s_m[i]
                t_s_m = self.t_s_m[i]
                slave_xyz = transform_vertices(rot_s_m, t_s_m, xyz)
                slave_dir = transform_direction(rot_s_m, view_dirs)
                slave_nabla = transform_direction(rot_s_m, nabla)
            else:
                slave_xyz, slave_dir = xyz, view_dirs
                slave_nabla = nabla
            if torch.any(paint_region) == True:
                if self.method == "code":
                    slave_color = slave_primitive.model.forward_editcolor(
                        self.main_primitive.edit_color_features,
                        _ds[paint_region],
                        slave_dir[paint_region],
                        indices=_indices[paint_region],
                        weights=slave_weights[paint_region],
                        nabla=slave_nabla[paint_region],
                    )
                elif self.method == "direct":
                    _, slave_color = slave_primitive.model(
                        slave_xyz[paint_region], slave_dir[paint_region]
                    )  # (...,3)
                else:
                    raise NotImplementedError
                if self.blend_black == False:
                    if self.blur_edge == True:
                        blend_color[paint_region] = blend_color[
                            paint_region
                        ] * unpaint_weight.unsqueeze(
                            -1
                        ) + slave_color * paint_weight.unsqueeze(
                            -1
                        )
                    else:
                        blend_color[paint_region] = slave_color
                else:
                    color = torch.zeros(3).to(blend_color.device)
                    color[i] = 1.0
                    blend_color[paint_region] = color

        # return (slave_sdf, slave_color)
        return (sdf, blend_color)
