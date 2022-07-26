import argparse
import open3d as o3d
import numpy as np
import json
from utils import io_util
from utils.dist_util import (
    get_local_rank,
)

from models.frameworks.neumesh import MeshSampler, get_neumesh_model as get_model
from models.frameworks.neumesh import SingleRenderer
from render import render_function, create_render_args
from tools import interactive_mesh_algnment
from models.edit import *
import torch


def normal_distribution(x, mu=0, sigma=1):
    return torch.exp(-0.5 * ((x - mu) ** 2) / (sigma ** 2)) / (
        sigma * np.sqrt(2 * np.pi)
    )  # normal distribution


def transfer_code_on_xyz(
    main_primitive,
    main_mask,
    slave_primitive,
    T_s_m,
    avg_method="inv_dis",
    debug_draw=False,
):
    # a. transform main vertices to slave space
    main_vertices = torch.FloatTensor(main_primitive.get_mesh_vertices(b_torch=False))
    slave_vertices = torch.FloatTensor(slave_primitive.get_mesh_vertices(b_torch=False))
    main_cand_vertices = main_vertices[main_mask, :]  # (Nm, 3)
    main_vertices_trans = transform_vertices(
        T_s_m[:3, :3], T_s_m[:3, 3], main_cand_vertices
    )  # (Nm,3)

    # b. compute closest 8 vertices for each main vertex
    _ds, _indices, _weights = slave_primitive.compute_distance_in_masked_region(
        main_vertices_trans.cuda()
    )  # (Nm, K, 1), (Nm, K), (Nm, K)
    _weights = _weights.detach().cpu()
    _indices = _indices.detach().cpu()

    # c. assign new code to main model
    Kc = 4

    close_indices = _indices[:, :Kc]  # (Nm, Kc)
    _raw_ds = torch.linalg.norm(
        main_vertices_trans.unsqueeze(-2) - slave_vertices[close_indices],
        dim=-1,
    )  # (Nm, K)
    if avg_method == "inv_dis":
        weights_t = 1 / (_raw_ds + 1e-7)  # (Nm, K)
    elif avg_method == "gaussian":
        sigma = (_raw_ds[:, 3] / (2.5)).unsqueeze(-1)  # (Nm, 1)
        weights_t = normal_distribution(_raw_ds, sigma=sigma)  # (Nm, K)
        print("use gaussian")
        pass

    weights_t = weights_t / torch.sum(weights_t, dim=-1, keepdims=True)  # (Nm , K)
    weights_t = weights_t.detach().cpu()

    feat_slave = slave_primitive.model.color_features[
        close_indices, :
    ]  # (Nm,Kc,fg_dim)
    feat_main_trans = torch.sum(
        weights_t.unsqueeze(-1) * feat_slave, dim=-2
    )  # (Nm, fg_dim)
    main_primitive.edit_color_features[main_mask] = feat_main_trans

    geo_feat_slave = slave_primitive.model.geometry_features[
        close_indices, :
    ]  # (Nm,Kc,fg_dim)
    geo_feat_main_trans = torch.sum(
        weights_t.unsqueeze(-1) * geo_feat_slave, dim=-2
    )  # (Nm, fg_dim)
    main_primitive.edit_geo_features[main_mask] = geo_feat_main_trans

    print(
        f"# feat whose closest Kc are old: {(np.any(slave_primitive.get_mask()[close_indices.numpy()], axis=-1) == False).sum()}"
    )
    # feat_main_trans[
    #     (torch.any(slave_primitive.mask[close_indices], dim=-1) == False)
    # ] = feat_0
    # # debug
    if debug_draw == True:
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(main_vertices_trans.cpu().numpy())
        pcd1.paint_uniform_color([1, 0.706, 0])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = slave_primitive.model.mesh_sampler.mesh.vertices
        pcd2.colors = slave_primitive.model.mesh_sampler.mesh.vertex_colors
        pcd_list = [pcd1, pcd2]
        l_pcd = o3d.geometry.LineSet()
        points_list = []
        points_indices = []
        for i in range(main_vertices_trans.shape[0]):
            v1 = main_vertices_trans[i].cpu().numpy()
            ind = close_indices[i, 0]
            v2 = slave_primitive.model.mesh_sampler.mesh_vertices[ind].cpu().numpy()
            points_list.append(v1[None, :])
            points_list.append(v2[None, :])
            points_indices.append([i * 2, i * 2 + 1])
        l_pcd.points = o3d.utility.Vector3dVector(
            np.concatenate(points_list, axis=0).astype(np.float32)
        )
        l_pcd.lines = o3d.utility.Vector2iVector(np.array(points_indices))
        l_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        pcd_list.append(l_pcd)
        o3d.visualization.draw_geometries(pcd_list)

    if debug_draw == True:
        mesh_main_debug = o3d.geometry.TriangleMesh()
        mesh_main = main_primitive.get_mesh()
        mesh_main_debug.vertices = mesh_main.vertices
        mesh_main_debug.triangles = mesh_main.triangles
        mesh_main_debug.vertex_colors = mesh_main.vertex_colors
        colors_main_debug = np.asarray(mesh_main_debug.vertex_colors)
        slave_colors = np.array(slave_primitive.get_mesh().vertex_colors)

        colors_searched = slave_colors[close_indices, :]  # (Nm,Kc,fg_dim)
        colors_brand = torch.sum(
            weights_t.unsqueeze(-1) * colors_searched, dim=-2
        )  # (Nm, fg_dim)
        colors_main_debug[main_mask] = colors_brand
        mesh_main_debug.vertex_colors = o3d.utility.Vector3dVector(colors_main_debug)
        o3d.visualization.draw_geometries([mesh_main_debug])


def vis_transformation(main_mesh, main_mask, slave_mesh, slave_mask, T_s_m):
    main_mesh_trans = o3d.geometry.TriangleMesh()
    main_mesh_trans.triangles = main_mesh.triangles
    main_color = np.array(main_mesh.vertex_colors)
    main_color[main_mask == False, :] = np.array([0, 0, 0])
    main_mesh_trans.vertex_colors = o3d.utility.Vector3dVector(main_color)

    vertices = np.asarray(main_mesh.vertices)
    vertices_trans = (T_s_m[:3, :3] @ vertices[..., None]).squeeze(-1) + T_s_m[:3, 3]
    main_mesh_trans.vertices = o3d.utility.Vector3dVector(vertices_trans)

    slave_mesh_trans = o3d.geometry.TriangleMesh()
    slave_mesh_trans.triangles = slave_mesh.triangles
    slave_mesh_trans.vertices = slave_mesh.vertices
    slave_color = np.asarray(slave_mesh.vertex_colors)
    slave_color[slave_mask == False, :] = np.array([0, 0, 0])
    slave_mesh_trans.vertex_colors = o3d.utility.Vector3dVector(slave_color)

    o3d.visualization.draw_geometries([slave_mesh_trans, main_mesh_trans])


def vis_mask_mesh(main_mesh, main_mask, slave_mesh, slave_mask, b_vis=True):
    main_mesh_trans = o3d.geometry.TriangleMesh()
    main_mesh_trans.triangles = main_mesh.triangles
    main_color = np.array(main_mesh.vertex_colors)
    main_color[main_mask == False, :] = 0
    main_mesh_trans.vertex_colors = o3d.utility.Vector3dVector(main_color)
    main_mesh_trans.vertices = main_mesh.vertices

    slave_mesh_trans = o3d.geometry.TriangleMesh()
    slave_mesh_trans.triangles = slave_mesh.triangles
    slave_mesh_trans.vertices = slave_mesh.vertices
    slave_color = np.array(slave_mesh.vertex_colors)
    slave_color[slave_mask == False, :] = 0
    slave_mesh_trans.vertex_colors = o3d.utility.Vector3dVector(slave_color)

    if b_vis == False:
        return main_mesh_trans, slave_mesh_trans
    else:
        o3d.visualization.draw_geometries([slave_mesh_trans, main_mesh_trans])


def check_degenerate_triangles(mesh):
    # check degenerate triangles
    vertices = np.asarray(mesh.vertices)
    for triangle in np.asarray(mesh.triangles):
        for j in range(3):
            i1 = triangle[j]
            i2 = triangle[(j + 1) % 3]
            v1 = vertices[i1]
            v2 = vertices[i2]
            dis = np.linalg.norm(v1 - v2)
            if dis < 1e-5:
                print(vertices[i1 : i2 + 1])
            assert (
                dis > 1e-5
            ), f"vertex {triangle[j]} and {triangle[(j + 1) % 3]} is too close: {dis}, v1: {v1}, v2:{v2}"


def interactive_rigid_transform_mesh(main_mesh, slave_mesh):
    main_pcd = o3d.geometry.PointCloud()
    main_pcd.points = main_mesh.vertices
    main_pcd.colors = main_mesh.vertex_colors
    slave_pcd = o3d.geometry.PointCloud()
    slave_pcd.points = slave_mesh.vertices
    slave_pcd.colors = slave_mesh.vertex_colors
    corr, T_s_m = interactive_mesh_algnment.demo_manual_registration(
        main_pcd, slave_pcd
    )
    return corr, T_s_m


def deform_mesh_func(pt1_trans, corr, slave_mesh, slave_mask, use_as_rigid_as_possible=True):
    # b. compute as rigid as possible
    if use_as_rigid_as_possible == True:
        check_degenerate_triangles(slave_mesh)
        vertices = np.asarray(slave_mesh.vertices)
        isolated_mask = get_isolated_mask(slave_mesh)
        static_ids = np.where(
            np.logical_or(slave_mask == False, isolated_mask == True)
        )[0]
        static_pos = np.array([vertices[i] for i in static_ids])
        handle_ids = corr
        handle_pos = pt1_trans

        if static_ids.shape[0] == 0:
            constraint_ids = o3d.utility.IntVector(handle_ids.astype(np.int32))
            constraint_pos = o3d.utility.Vector3dVector(handle_pos)
        else:
            constraint_ids = o3d.utility.IntVector(
                np.concatenate([static_ids, handle_ids], axis=0).astype(np.int32)
            )
            constraint_pos = o3d.utility.Vector3dVector(
                np.concatenate([static_pos, handle_pos], axis=0)
            )
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            mesh_prime = slave_mesh.deform_as_rigid_as_possible(
                constraint_ids, constraint_pos, max_iter=20
            )
        slave_mesh.vertices = mesh_prime.vertices


def get_isolated_mask(mesh):
    triangles = np.asarray(mesh.triangles)

    mask = np.zeros(np.asarray(mesh.vertices).shape[0])
    used_vertices = triangles.flatten()
    mask[used_vertices] = 1
    return mask == 0


def clean_duplicate_triangles(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    v0 = vertices[triangles[:, 0]]  # (M,3)
    v1 = vertices[triangles[:, 1]]  # (M,3)
    v2 = vertices[triangles[:, 2]]  # (M,3)

    dis1 = np.linalg.norm(v0 - v1, axis=-1)  # (M)
    dis2 = np.linalg.norm(v1 - v2, axis=-1)  # (M)
    dis3 = np.linalg.norm(v2 - v0, axis=-1)  # (M)

    valid = (dis1 > 1e-5) & (dis2 > 1e-5) & (dis3 > 1e-5)
    triangles = triangles[valid, :]
    mesh.triangles = o3d.utility.Vector3iVector(triangles)


def read_data(main_config, mask_paths, ckpt_file, device):
    # Create main model
    main_args = io_util.load_yaml(main_config)
    (
        model,
        render_kwargs_test,
        _,
    ) = get_model(main_args)
    state_dict = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state_dict["model"])
    masks = []
    for mask_path in mask_paths:
        mask_mesh = o3d.io.read_triangle_mesh(mask_path)
        mask = np.sum(np.asarray(mask_mesh.vertex_colors), axis=-1) != 0
        masks.append(mask)
        print("number_of_vertices_paint: {}".format(mask.sum()))
    primitive = EditablePrimitive(
        model,
        masks,
        None,
        None,
        color_feature=model.color_features,
        geo_feature=model.geometry_features,
    )
    return (
        primitive,
        main_args,
        render_kwargs_test,
    )  # type(main_args): utils.io_util.ForceKeyErrorDict


def save_rigid_transform(args, T_s_m_list, corr_list):
    with open(args.config, "r") as f:
        data = json.loads(f.read())
        data["T_s_m"] = np.array(T_s_m_list).tolist()
        data["corr"] = [a.tolist() for a in corr_list]
    with open(args.config, "w") as f:
        json.dump(data, f, indent=2)


def main_function(args):

    # init_env(args)

    # # ----------------------------
    # # -------- shortcuts ---------
    # rank = get_rank()
    local_rank = get_local_rank()

    device = torch.device("cuda", local_rank)

    # a. read data

    main_primitive, main_args, render_kwargs_test = read_data(
        args.main_config, args.main_mask_mesh, args.main_ckpt, args.device
    )

    slave_primitives = []
    for i in range(len(args.slave_config)):
        primtive, _, _ = read_data(
            args.slave_config[i],
            [args.slave_mask_mesh[i]],
            args.slave_ckpt[i],
            args.device,
        )
        slave_primitives.append(primtive)

    assert main_primitive.get_len_of_mask() == len(
        slave_primitives
    ), "Error: the number of main mask is not mached with number of slave objects"

    # b. compute srt
    T_s_m_list = []
    corr_list = []
    for i, slave_primitive in enumerate(slave_primitives):
        if i in args.estimate_srt:
            main_mask_mesh = main_primitive.get_masked_mesh(i)
            slave_mask_mesh = slave_primitive.get_masked_mesh()
            corr, T_s_m = interactive_rigid_transform_mesh(
                main_mask_mesh, slave_mask_mesh
            )
            corr = np.int32(corr)
        else:
            T_s_m = np.array(args.T_s_m[i])
            corr = np.array(args.corr[i])
        T_s_m_list.append(T_s_m)
        corr_list.append(corr)
    if args.debug_draw == True:
        save_rigid_transform(args, T_s_m_list, corr_list)

    # c. deform mesh
    if args.rigid_transform == False:
        for i, slave_primitive in enumerate(slave_primitives):
            T_s_m = T_s_m_list[i]
            corr = corr_list[i]
            slave_mesh = slave_primitive.get_mesh()
            slave_distance_method = slave_primitive.get_mesh_sampler().distance_method
            slave_minimum_box = slave_primitive.get_mesh_sampler().minimum_box
            clean_duplicate_triangles(slave_mesh)
            pt1 = main_primitive.get_mesh_vertices(b_torch=False)[corr[:, 0]]
            pt1_trans = (T_s_m[:3, :3] @ pt1[..., None]).squeeze(-1) + T_s_m[:3, 3]
            deform_mesh_func(
                pt1_trans,
                corr[:, 1],
                slave_mesh,
                slave_primitive.get_mask(),
                use_as_rigid_as_possible=True,
            )
            slave_primitive.model.mesh_sampler = MeshSampler(
                slave_mesh,
                device,
                distance_method=slave_distance_method,
                minimum_box=slave_minimum_box,
            )
            if args.debug_draw:
                vis_transformation(
                    main_primitive.get_mesh(),
                    main_primitive.get_mask(i),
                    slave_primitive.get_mesh(),
                    slave_primitive.get_mask(),
                    np.array(T_s_m),
                )

    # d. transfer code
    for i in range(len(T_s_m_list)):
        T_s_m_list[i] = torch.FloatTensor(T_s_m_list[i])
        transfer_code_on_xyz(
            main_primitive,
            main_primitive.get_mask(i),
            slave_primitives[i],
            T_s_m_list[i],
            debug_draw=args.debug_draw,
        )
    if args.debug_draw == True:
        return
    # c. deform mesh
    if "deform_mesh" in args and args.deform_mesh is not None:
        # print("deform mesh")
        # deform_mesh = o3d.io.read_triangle_mesh(args.deform_mesh)
        # transform_model(deform_mesh, main_primitive.model, device, fix_indicator=args.fix_indicator)
        pass
    # d. create TextureEditableNeuMesh
    main_primitive.to_tensor(device)
    for i, slave_primitive in enumerate(slave_primitives):
        slave_primitive.to_tensor(device)
        T_s_m_list[i] = T_s_m_list[i].to(device)
    model = TextureEditableNeuMesh(
        main_primitive,
        slave_primitives,
        T_s_m_list,
        method=args.edit_method,
        blur_edge=True,
    )

    model.to(device).eval()

    # d. render_view
    renderer = SingleRenderer(model)
    args.update(dict(main_args))
    render_kwargs_test["white_bkgd"] = True
    render_function(args, model, render_kwargs_test, renderer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    create_render_args(parser)
    parser.add_argument("--edit_method", type=str, default="code")
    parser.add_argument("--blend_black", action="store_true", default=False)
    parser.add_argument("--rigid_transform", action="store_true", default=False)
    parser.add_argument("--debug_draw", action="store_true")
    parser.add_argument("--estimate_srt", nargs="+", default=[], type=int)
    args, unknown = parser.parse_known_args()
    config_dict = io_util.read_json(args.config)
    other_dict = vars(args)
    config_dict.update(other_dict)
    config = io_util.ForceKeyErrorDict(**config_dict)
    main_function(config)
