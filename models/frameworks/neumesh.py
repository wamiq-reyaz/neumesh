import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
from torch import autograd

from utils import rend_util, train_util
from utils.metric_util import *

import functools
import contextlib
import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict
import open3d as o3d

from models.base import get_embedder
import frnn
from utils.checkpoints import CheckpointIO


def load_ckpt(ckpt_path, model):
    checkpoint_io = CheckpointIO(allow_mkdir=False)
    checkpoint_io.register_modules(model=model)
    checkpoint_io.load_file(ckpt_path)

def cdf_Phi_s(x, s):  # \VarPhi_s(t)
    # den = 1 + torch.exp(-s*x)
    # y = 1./den
    # return y
    return torch.sigmoid(x * s)


def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha


def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    cdf, opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ],
        dim=-1,
    )

    # [(B), N_rays, N_pts-1]
    visibility_weights = (
        opacity_alpha * torch.cumprod(shifted_transparency, dim=-1)[..., :-1]
    )

    return cdf, opacity_alpha, visibility_weights


def alpha_to_w(alpha: torch.Tensor):
    device = alpha.device
    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*alpha.shape[:-1], 1], device=device),
            1.0 - alpha + 1e-10,
        ],
        dim=-1,
    )

    # [(B), N_rays, N_pts-1]
    visibility_weights = alpha * torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return visibility_weights


class NeuMesh(nn.Module):
    def __init__(
        self,
        mesh_sampler,
        b_xyz_input,
        D_density: int,
        D_color: int,
        W: int,
        geometry_dim: int,
        color_dim: int,
        multires_view: int,
        multires_d: int,
        multires_fg: int,
        multires_ft: int,
        enable_nablas_input: bool,
        input_view_dim=3,
        input_d_dim=1,
        ln_s=0.2996,
        speed_factor=1.0,
        learn_indicator_vector=False,
        learn_indicator_weight=True,
        enable_geometry_emb=True,
        enable_color_emb=True,
        enable_geometry_input=False,
    ):
        super(NeuMesh, self).__init__()

        self.mesh_sampler = mesh_sampler
        self.b_xyz_input = b_xyz_input
        num_vertices = np.asarray(self.mesh_sampler.mesh.vertices).shape[0]

        self.ln_s = nn.Parameter(torch.Tensor([ln_s]), requires_grad=True)
        self.speed_factor = speed_factor

        self.geometry_features = nn.Parameter(
            torch.randn(num_vertices, geometry_dim, dtype=torch.float32)
        )
        self.color_features = nn.Parameter(
            torch.randn(num_vertices, color_dim, dtype=torch.float32)
        )

        # indicator vector for frnn projected distance
        self.learn_indicator_vector = learn_indicator_vector
        if learn_indicator_vector:
            indicator_vector = self.mesh_sampler.vertex_normals.float().clone()
            self.indicator_vector = nn.Parameter(indicator_vector)

        # indicator weight for frnn projected distance
        self.learn_indicator_weight = learn_indicator_weight
        if self.learn_indicator_weight:
            self.indicator_weight_raw = nn.Parameter(
                torch.Tensor([-2]), requires_grad=True
            )

        self.embed_fn_d, input_ch_d = get_embedder(multires_d, input_dim=input_d_dim)
        self.embed_fn_view, input_ch_view = get_embedder(
            multires_view, input_dim=input_view_dim
        )
        if enable_geometry_emb == True:
            self.embed_fn_fg, input_ch_fg = get_embedder(
                multires_fg, input_dim=geometry_dim
            )
            input_ch_pts = input_ch_d + input_ch_fg
            print(f"geometry_dim: {geometry_dim}, input_ch_fg: {input_ch_fg}")
        else:
            input_ch_pts = input_ch_d

        if enable_color_emb == True:
            self.embed_fn_ft, input_ch_ft = get_embedder(
                multires_ft, input_dim=color_dim
            )
            input_ch_color = input_ch_view + input_ch_ft + input_ch_d
            print(f"color_dim: {color_dim}, input_ch_ft: {input_ch_ft}")

        else:
            input_ch_color = input_ch_view

        if enable_geometry_input == True:
            input_ch_color += W
        self.enable_geometry_input = enable_geometry_input

        self.enable_nablas_input = enable_nablas_input
        if self.enable_nablas_input:
            input_ch_color += 3

        self.enable_geometry_emb = enable_geometry_emb
        self.enable_color_emb = enable_color_emb
        self.softplus = nn.Softplus(beta=100)
        self.pts_linears = nn.Sequential(
            weight_norm(nn.Linear(input_ch_pts, W)),
            # nn.ReLU(inplace=True),
            self.softplus,
            *[
                nn.Sequential(
                    weight_norm(nn.Linear(W, W)),
                    # nn.ReLU(inplace=True),
                    self.softplus,
                )
                for i in range(D_density - 1)
            ],
        )
        self.views_linears = nn.Sequential(
            nn.Linear(input_ch_color, W),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True))
                for i in range(D_color - 1)
            ],
        )
        self.density_linear = weight_norm(nn.Linear(W, 1))
        self.color_linear = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())

        print(
            f"input_d_dim: {input_d_dim}, input_view_dim: {input_view_dim}, input_ch_d: {input_ch_d}, input_ch_view: {input_ch_view}, input_ch_pts: {input_ch_pts}, input_ch_color: {input_ch_color}"
        )


    def compute_distance(self, xyz):
        _ds, _indices, _weights = self.mesh_sampler.compute_distance(
            # xyz.detach().view(-1, 3),
            xyz.view(-1, 3),
            indicator_vector=self.indicator_vector
            if self.learn_indicator_vector
            else None,
            indicator_weight=self.forward_indicator_weight()
            if self.learn_indicator_weight
            else 0.1,
        )
        _ds = _ds.reshape(*xyz.shape[:-1], -1)
        _indices = _indices.reshape(*xyz.shape[:-1], -1)
        _weights = _weights.reshape(*xyz.shape[:-1], -1)
        return _ds, _indices, _weights

    def forward(
        self,
        xyz: torch.Tensor,
        view_dirs: torch.Tensor,
        density_only=False,
        need_nablas=True,
        nablas_only=False,
    ):
        if self.b_xyz_input == True:
            out = self._forward(xyz, view_dirs, density_only=density_only)
        else:
            if need_nablas:
                xyz.requires_grad_(True)
            with (torch.enable_grad() if need_nablas else contextlib.nullcontext()):
                _ds, _indices, _weights = self.compute_distance(xyz)
            out = self._forward(
                xyz,
                _ds,
                view_dirs,
                _indices,
                _weights,
                density_only=density_only,
                need_nablas=need_nablas,
                nablas_only_for_eikonal=nablas_only,
            )

        return out

    def forward_density_only(self, xyz, view_dirs):
        return self.forward(xyz, view_dirs, density_only=True, need_nablas=False)

    def forward_with_nablas(self, xyz: torch.Tensor, view_dirs: torch.Tensor):
        return self.forward(
            xyz,
            view_dirs,
            density_only=False,
            need_nablas=True,
            nablas_only=True,
        )

    def interpolation(
        self, features: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ):
        fg = torch.sum(features[indices] * weights.unsqueeze(-1), dim=-2)
        return fg

    def _forward(
        self,
        xyz: torch.Tensor,
        d: torch.Tensor,
        view_dirs: torch.Tensor,
        indices: torch.Tensor = None,
        weights: torch.Tensor = None,
        density_only: bool = False,
        need_nablas: bool = False,
        nablas_only_for_eikonal: bool = False,
        # skip_far_points: bool = True,
        skip_far_points: bool = False,
        far_point_thresh: float = 0.1,
    ):
        """
        d: (N,1), distance from point to nearest mesh
        view_dirs: (N,3)
        indices: (N, 3)
        weights: (N, 3)
        """
        if len(d.shape) != 3:
            # not compatible when not in training mode
            skip_far_points = False
        if skip_far_points:
            N_full = d.shape[1]
            mask = d.squeeze().abs() < far_point_thresh
            ind_full = torch.arange(N_full)
            ind_occu = ind_full[mask]
            d = d[:, ind_occu]
            view_dirs = view_dirs[:, ind_occu]
            indices = indices[:, ind_occu]
            weights = weights[:, ind_occu]

        if need_nablas:
            assert not skip_far_points, "not implemented"

        with (torch.enable_grad() if need_nablas else contextlib.nullcontext()):
            d_emb = self.embed_fn_d(d)
            view_dirs_emb = self.embed_fn_view(view_dirs)
            if self.enable_geometry_emb == True:
                fg = self.interpolation(self.geometry_features, indices, weights)
                fg_emb = self.embed_fn_fg(fg)
                h = self.pts_linears(torch.cat([d_emb, fg_emb], dim=-1))
            else:
                h = self.pts_linears(torch.cat([d_emb], dim=-1))
            density = self.density_linear(h)

        if density_only:
            return density, torch.zeros_like(density)

        has_grad = torch.is_grad_enabled()
        if need_nablas:
            nabla = autograd.grad(
                density,
                xyz,
                torch.ones_like(density, device=xyz.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True,
            )[0]

        if not has_grad:
            nabla = nabla.detach()

        if nablas_only_for_eikonal:
            return density, nabla

        color_input = []
        if self.enable_geometry_input == True:
            color_input.append(h)
        if self.enable_nablas_input:
            color_input.append(nabla)
        color_input.append(d_emb)
        color_input.append(view_dirs_emb)
        if self.enable_color_emb == True:
            ft = self.interpolation(self.color_features, indices, weights)
            ft_emb = self.embed_fn_ft(ft)
            color_input.append(ft_emb)

        hv = self.views_linears(torch.cat(color_input, dim=-1))  # [(h), dir_emb, ft]
        color = self.color_linear(hv)

        if skip_far_points:
            density_full = -torch.ones((1, N_full, 1), device=density.device)
            density_full[:, ind_occu] = density
            color_full = torch.zeros((1, N_full, 3), device=density.device)
            color_full[:, ind_occu] = color
            return density_full, color_full

        return density, color

    def forward_with_ds(
        self,
        xyz: torch.Tensor,
        view_dirs: torch.Tensor,
        density_only=False,
        need_nablas=True,
        nablas_only=False,
    ):
        if self.b_xyz_input == True:
            out = self._forward(xyz, view_dirs, density_only=density_only)
        else:
            if need_nablas:
                xyz.requires_grad_(True)
            with (torch.enable_grad() if need_nablas else contextlib.nullcontext()):
                _ds, _indices, _weights = self.compute_distance(xyz)
            out = self._forward(
                xyz,
                _ds,
                view_dirs,
                _indices,
                _weights,
                density_only=density_only,
                need_nablas=need_nablas,
                nablas_only_for_eikonal=nablas_only,
            )
            out = out + (_ds, _indices, _weights)

        return out

    def forward_editcolor(
        self,
        color_features,
        d: torch.Tensor,
        view_dirs: torch.Tensor,
        indices: torch.Tensor = None,
        weights: torch.Tensor = None,
        h=None,
        nabla=None,
    ):

        d_emb = self.embed_fn_d(d)
        view_dirs_emb = self.embed_fn_view(view_dirs)
        color_input = []
        if self.enable_geometry_input == True:
            color_input.append(h)
        if self.enable_nablas_input:
            color_input.append(nabla)
        color_input.append(d_emb)
        color_input.append(view_dirs_emb)
        if self.enable_color_emb == True:
            ft = self.interpolation(color_features, indices, weights)
            ft_emb = self.embed_fn_ft(ft)
            color_input.append(ft_emb)

        hv = self.views_linears(torch.cat(color_input, dim=-1))  # [(h), dir_emb, ft]
        color = self.color_linear(hv)

        return color

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)

    def forward_indicator_weight(self):
        return torch.sigmoid(self.indicator_weight_raw)

    def compute_bounded_near_far(
        self,
        rays_o,
        rays_d,
        near,
        far,
        sample_grid: int = 256,
        distance_thresh: float = 0.1,
    ):
        near_orig = near.clone()
        far_orig = far.clone()
        # rays_o, rays_d: (1, N_rays, 3)
        # near, far: (1, N_rays, 1)
        _t = torch.linspace(0, 1, sample_grid, device=rays_o.device)
        # d_coarse: (1, N_rays, N_grid)
        d_coarse = near * (1 - _t) + far * _t
        # d_coarse: (1, N_rays, N_grid, 1)
        d_coarse = d_coarse.unsqueeze(-1)
        # pts_coarse: (1, N_rays, N_grid, 3)
        pts_coarse = rays_o.unsqueeze(-2) + d_coarse * rays_d.unsqueeze(-2)
        ds, _, _ = self.compute_distance(pts_coarse)
        mask = ds < distance_thresh

        near = d_coarse * mask.float() + (~mask).float() * 1e10
        near = near.min(dim=-2, keepdim=False)[0]
        near_mask = near > 1e5
        near[near_mask] = near_orig[near_mask]

        far = d_coarse * mask.float() - (~mask).float() * 1e10
        far = far.max(dim=-2, keepdim=False)[0]
        far_mask = far < -1e5
        far[far_mask] = far_orig[far_mask]
        # compensate too small near far
        too_close = (far - near) < 0.1
        far[too_close] += 0.05
        near[too_close] -= 0.05
        return near, far


def volume_render(
    rays_o,
    rays_d,
    model: NeuMesh,
    obj_bounding_radius=1.0,
    batched=False,
    batched_info={},
    # render algorithm config
    calc_normal=False,
    use_view_dirs=True,
    rayschunk=65536,
    netchunk=1048576,
    white_bkgd=False,
    near_bypass: Optional[float] = None,
    far_bypass: Optional[float] = None,
    # render function config
    detailed_output=True,
    show_progress=False,
    # sampling related
    perturb=False,  # config whether do stratified sampling
    fixed_s_recp=1 / 64.0,
    N_samples=64,
    N_importance=64,
    N_outside=0,  # whether to use outside nerf
    # upsample related
    upsample_algo="official_solution",
    N_nograd_samples=2048,
    N_upsample_iters=4,
    b_out_samples=False,
    compute_bounded_near_far=True,
    random_color_direction=False,
    **dummy_kwargs,  # just place holder
):
    """
    input:
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3] NOTE: not normalized. contains info about ratio of len(this ray)/len(principle ray)
    """
    device = rays_o.device
    if batched:
        DIM_BATCHIFY = 1
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_BATCHIFY = 0
        flat_vec_shape = [-1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()
    # NOTE: already normalized
    rays_d = F.normalize(rays_d, dim=-1)

    batchify_query = functools.partial(
        train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY
    )

    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]

        # [(B), N_rays] x 2
        near, far = rend_util.near_far_from_sphere(
            rays_o, rays_d, r=obj_bounding_radius
        )   
        if compute_bounded_near_far:
            near, far = model.compute_bounded_near_far(rays_o, rays_d, near, far)
        if near_bypass is not None:
            near = near_bypass * torch.ones_like(near).to(device)
        if far_bypass is not None:
            far = far_bypass * torch.ones_like(far).to(device)

        if use_view_dirs:
            view_dirs = rays_d
        else:
            view_dirs = None

        prefix_batch = [B] if batched else []
        N_rays = rays_o.shape[-2]

        # ---------------
        # Sample points on the rays
        # ---------------

        # ---------------
        # Coarse Points

        # [(B), N_rays, N_samples]
        _t = torch.linspace(0, 1, N_samples).float().to(device)
        d_coarse = near * (1 - _t) + far * _t

        # ---------------
        # Up Sampling
        if b_out_samples == True:
            samples_tot = {"xyz": [], "density": [], "colors": []}
        with torch.no_grad():
            if upsample_algo == "official_solution":
                _d = d_coarse
                _xyz = rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2)
                _sdf, _ = batchify_query(
                    model.forward_density_only,
                    _xyz,
                    view_dirs.unsqueeze(-2).expand_as(_xyz),
                )
                _sdf = _sdf.squeeze(-1)
                for i in range(N_upsample_iters):
                    prev_sdf, next_sdf = (
                        _sdf[..., :-1],
                        _sdf[..., 1:],
                    )  # (...,N_samples-1)
                    prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (
                        next_z_vals - prev_z_vals + 1e-5
                    )  
                    prev_dot_val = torch.cat(
                        [
                            torch.zeros_like(dot_val[..., :1], device=device),
                            dot_val[..., :-1],
                        ],
                        dim=-1,
                    )  # jianfei: prev_slope, right shifted,  (...,N_samples-1)
                    dot_val = torch.stack(
                        [prev_dot_val, dot_val], dim=-1
                    )  # jianfei: concat prev_slope with slope ,  (...,N_samples-1,2)
                    dot_val, _ = torch.min(
                        dot_val, dim=-1, keepdim=False
                    )  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope),  (...,N_samples-1)
                    dot_val = dot_val.clamp(-10.0, 0.0)

                    dist = next_z_vals - prev_z_vals
                    prev_esti_sdf = (
                        mid_sdf - dot_val * dist * 0.5
                    ) 
                    next_esti_sdf = mid_sdf + dot_val * dist * 0.5

                    # phi_s_base = 64
                    phi_s_base = 256
                    prev_cdf = cdf_Phi_s(
                        prev_esti_sdf, phi_s_base * (2 ** i)
                    )  # \VarPhi_s(x) in paper
                    next_cdf = cdf_Phi_s(next_esti_sdf, phi_s_base * (2 ** i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                    _w = alpha_to_w(alpha)
                    d_fine = rend_util.sample_pdf(
                        _d, _w, N_importance // N_upsample_iters, det=not perturb
                    )
                    _d = torch.cat([_d, d_fine], dim=-1)

                    pts_fine = rays_o.unsqueeze(-2) + d_fine.unsqueeze(
                        -1
                    ) * rays_d.unsqueeze(-2)
                    sdf_fine, _ = batchify_query(
                        model.forward_density_only,
                        pts_fine,
                        view_dirs.unsqueeze(-2).expand_as(pts_fine),
                    )
                    sdf_fine = sdf_fine.squeeze(-1)
                    _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                    _d, d_sort_indices = torch.sort(_d, dim=-1)
                    _sdf = torch.gather(_sdf, DIM_BATCHIFY + 1, d_sort_indices)
                d_all = _d
            else:
                raise NotImplementedError

        # ------------------
        # Calculate Points
        # [(B), N_rays, N_samples+N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        # [(B), N_rays, N_pts-1, 3]
        # pts_mid = 0.5 * (pts[..., 1:, :] + pts[..., :-1, :])
        d_mid = 0.5 * (d_all[..., 1:] + d_all[..., :-1])
        pts_mid = rays_o[..., None, :] + rays_d[..., None, :] * d_mid[..., :, None]
        # ------------------
        # Inside Scene
        # ------------------
        if calc_normal:
            sdf, nablas = batchify_query(
                model.forward_with_nablas, pts, view_dirs.unsqueeze(-2).expand_as(pts)
            )
        else:
            sdf, _ = batchify_query(
                model.forward_density_only, pts, view_dirs.unsqueeze(-2).expand_as(pts)
            )

        sdf = sdf.squeeze(-1)
        # [(B), N_ryas, N_pts], [(B), N_ryas, N_pts-1]
        cdf, opacity_alpha = sdf_to_alpha(
            sdf, model.forward_s()
        )   
        # radiances = model.forward_radiance(pts_mid, view_dirs_mid)
        if random_color_direction == False:
            rad_sdf, radiances = batchify_query(
                model.forward, pts_mid, view_dirs.unsqueeze(-2).expand_as(pts_mid)
            )
        else:
            random_direction = torch.rand_like(pts_mid)
            random_direction /= torch.linalg.norm(
                random_direction, axis=-1, keepdims=True
            )
            rad_sdf, radiances = batchify_query(
                model.forward, pts_mid, random_direction
            )
        if b_out_samples == True:
            samples_tot["xyz"].append(pts_mid)
            samples_tot["density"].append(rad_sdf)
            samples_tot["colors"].append(radiances)

        # --------------
        # Ray Integration
        # --------------
        d_final = d_mid

        # [(B), N_ryas, N_pts-1 + N_outside]
        visibility_weights = alpha_to_w(opacity_alpha)
        # [(B), N_rays]
        rgb_map = torch.sum(visibility_weights[..., None] * radiances, -2)
        # depth_map = torch.sum(visibility_weights * d_mid, -1)
        # NOTE: to get the correct depth map, the sum of weights must be 1!
        depth_map = torch.sum(
            visibility_weights
            / (visibility_weights.sum(-1, keepdim=True) + 1e-10)
            * d_final,
            -1,
        )
        acc_map = torch.sum(visibility_weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict(
            [
                ("rgb", rgb_map),  # [(B), N_rays, 3]
                ("depth_volume", depth_map),  # [(B), N_rays]
                # ('depth_surface', d_pred_out),    # [(B), N_rays]
                ("mask_volume", acc_map),  # [(B), N_rays]
            ]
        )

        if calc_normal:
            normals_map = F.normalize(nablas, dim=-1)
            N_pts = min(visibility_weights.shape[-1], normals_map.shape[-2])
            normals_map = (
                normals_map[..., :N_pts, :] * visibility_weights[..., :N_pts, None]
            ).sum(dim=-2)
            ret_i["normals_volume"] = normals_map

        if detailed_output:
            if calc_normal:
                ret_i["implicit_nablas"] = nablas
            ret_i["implicit_surface"] = sdf
            ret_i["radiance"] = radiances
            ret_i["alpha"] = opacity_alpha
            ret_i["cdf"] = cdf
            ret_i["visibility_weights"] = visibility_weights
            ret_i["d_final"] = d_final
            if N_outside > 0:
                # ret_i['sigma_out'] = sigma_out
                # ret_i['radiance_out'] = radiance_out
                pass
            if b_out_samples == True:
                ret_i["xyz"] = torch.cat(samples_tot["xyz"], 2)
                ret_i["dirs"] = view_dirs.unsqueeze(-2).expand_as(ret_i["xyz"])
                ret_i["density"] = torch.cat(samples_tot["density"], 2)
                ret_i["colors"] = torch.cat(samples_tot["colors"], 2)

        return ret_i

    ret = {}
    for i in tqdm(
        range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress
    ):
        ret_i = render_rayschunk(
            rays_o[:, i : i + rayschunk] if batched else rays_o[i : i + rayschunk],
            rays_d[:, i : i + rayschunk] if batched else rays_d[i : i + rayschunk],
        )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)

    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)

    return ret["rgb"], ret["depth_volume"], ret


class MeshSampler:
    def __init__(self, mesh, device, distance_method, minimum_box):
        # Note: the modification of self.mesh will not affect the compute_distance_(o3d/frnn)
        self.mesh = mesh
        # self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        _ = self.scene.add_triangles(mesh_t)  # we do not need the geometry ID for mesh

        self.mesh_vertices = torch.FloatTensor(np.asarray(self.mesh.vertices)).to(
            device
        )
        self.vertex_normals = torch.FloatTensor(
            np.asarray(self.mesh.vertex_normals)
        ).to(device)
        _, _, _, self.grid = frnn.frnn_grid_points(
            self.mesh_vertices.unsqueeze(0),
            self.mesh_vertices.unsqueeze(0),
            None,
            None,
            K=32,
            r=100.0,
            grid=None,
            return_nn=False,
            return_sorted=True,
        )

        self.distance_method = distance_method
        self.minimum_box = minimum_box
        print("NeuMesh distance method " + str(distance_method))

        # verbose:

    def compute_distance(self, xyz, indicator_vector=None, indicator_weight=0.1):
        method = self.distance_method
        if method == "frnn":
            return self.compute_distance_frnn(
                xyz,
                use_middle_vector=True,
                indicator_vector=indicator_vector,
                indicator_weight=indicator_weight,
            )
        else:
            raise NotImplementedError

    def compute_distance_frnn(
        self,
        xyz,
        K=8,
        signed_distance=True,
        use_middle_vector=False,
        indicator_vector=None,
        indicator_weight=0.1,
    ):
        dis, indices, _, _ = frnn.frnn_grid_points(
            xyz.unsqueeze(0),
            self.mesh_vertices.unsqueeze(0),
            None,
            None,
            K=K,
            r=100.0,
            grid=self.grid,
            return_nn=False,
            return_sorted=True,
        )  # (1,M,K)
        # detach to make the other differentiable
        dis = dis.detach()
        indices = indices.detach()
        dis = dis.sqrt()
        weights = 1 / (dis + 1e-7)
        weights = weights / torch.sum(weights, dim=-1, keepdims=True)  # (1,M,K)
        indices = indices.squeeze(0)  # (M, K)
        weights = weights.squeeze(0)  # (M, K)
        distance = torch.zeros((xyz.shape[0], 1)).to(xyz.device)
        if use_middle_vector:
            indicator_vec = (
                self.vertex_normals if indicator_vector is None else indicator_vector
            )
        if signed_distance:
            if use_middle_vector:
                w1 = indicator_weight
                dir_vec = xyz.unsqueeze(-2) - self.mesh_vertices[indices]
                w2 = torch.norm(dir_vec, dim=-1, keepdim=True)
                middle_vec = (indicator_vec[indices] * w1 + dir_vec * w2) / (w1 + w2)
                distance = weights.unsqueeze(-1) * torch.sum(
                    dir_vec * middle_vec,
                    dim=-1,
                    keepdim=True,
                )
                distance = torch.sum(distance, dim=-2)
            else:
                distance = weights.unsqueeze(-1) * torch.sum(
                    (xyz.unsqueeze(-2) - self.mesh_vertices[indices])
                    * self.vertex_normals[indices],
                    dim=-1,
                    keepdim=True,
                )
                distance = torch.sum(distance, dim=-2)
        else:
            distance = weights.unsqueeze(-1) * torch.norm(
                xyz.unsqueeze(-2) - self.mesh_vertices[indices], dim=-1
            )

            distance = torch.sum(distance, dim=-2)
        # distance = dis[..., 0:1]
        return distance, indices, weights

    def cast_ray(self, rays_o, rays_d):
        rays_core = np.concatenate([rays_o, rays_d], axis=1)
        rays = o3d.core.Tensor([rays_core], dtype=o3d.core.Dtype.Float32)
        ans = self.scene.cast_rays(rays)
        return ans["t_hit"].numpy(), ans["primitive_ids"].numpy()


class SingleRenderer(nn.Module):
    def __init__(self, model: NeuMesh):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)

def get_neumesh_model(args):

    mesh = o3d.io.read_triangle_mesh(args.neumesh.prior_mesh)
    mesh_sampler = MeshSampler(
        mesh,
        args.device_ids[0],
        args.neumesh.setdefault("distance_method", "frnn"),
        minimum_box=args.neumesh.setdefault("minimum_sampleing_box", 0),
    )

    # placeholder for neumesh keys
    if "neumesh" not in args.model:
        args.model["neumesh"] = {}

    model_config = {
        "speed_factor": args.training.setdefault("speed_factor", 1.0),
        "D_density": args.model.neumesh.setdefault("D_density", 3),
        "D_color": args.model.neumesh.setdefault("D_color", 4),
        "W": args.model.neumesh.setdefault("W", 256),
        "geometry_dim": args.model.neumesh.setdefault("geometry_dim", 32),
        "color_dim": args.model.neumesh.setdefault("color_dim", 32),
        "multires_view": args.model.neumesh.setdefault("multires_view", 4),
        "multires_d": args.model.neumesh.setdefault("multires_d", 8),
        "multires_fg": args.model.neumesh.setdefault("multires_fg", 2),
        "multires_ft": args.model.neumesh.setdefault("multires_ft", 2),
        "enable_nablas_input": args.neumesh.setdefault("enable_nablas_input", False),
    }

    ## render kwargs
    render_kwargs_test = {
        # upsample config
        "upsample_algo": args.model.setdefault(
            "upsample_algo", "official_solution"
        ),  # [official_solution, direct_more, direct_use]
        "N_nograd_samples": args.model.setdefault("N_nograd_samples", 2048),
        "N_upsample_iters": args.model.setdefault("N_upsample_iters", 4),
        "N_outside": args.model.setdefault("N_outside", 0),
        "obj_bounding_radius": args.data.setdefault("obj_bounding_radius", 1.0),
        "batched": args.data.batch_size is not None,
        "perturb": args.model.setdefault(
            "perturb", True
        ),  # config whether do stratified sampling
        "white_bkgd": args.model.setdefault("white_bkgd", False),
        "compute_bounded_near_far": args.model.setdefault(
            "compute_bounded_near_far", True
        ),
    }

    if args.neumesh.input_xyz == True:
        model_config["input_d_dim"] = 3
        model_config["multires_d"] = 6
        model_config["enable_geometry_emb"] = False
        model_config["enable_color_emb"] = False
        model_config["enable_geometry_input"] = True
        render_kwargs_test["b_xyz_input"] = True
    else:
        render_kwargs_test["b_xyz_input"] = False

    model_config["learn_indicator_vector"] = args.neumesh.get(
        "learn_indicator_vector", False
    )
    model_config["learn_indicator_weight"] = args.neumesh.get(
        "learn_indicator_weight", False
    )


    has_eikonal = float(args.neumesh.loss_weights.setdefault("eikonal", 0.0))

    if has_eikonal > 0:
        render_kwargs_test["calc_normal"] = True

    render_kwargs_test["rayschunk"] = args.data.val_rayschunk
    render_kwargs_test["perturb"] = False

    model = NeuMesh(mesh_sampler, args.neumesh.input_xyz, **model_config)

    renderer = SingleRenderer(model)

    return model, render_kwargs_test, renderer
