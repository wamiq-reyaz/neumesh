""" Author: Wamiq Reyaz Para
Date: 05 October 2022
Description: Utilities to save barycentric maps and face indices for meshes into 
.pkl or numpy or similar formats.
"""

import argparse
import os
import sys
import pickle
import json

import torch
import numpy as np
from tqdm import tqdm
import trimesh as trim
curr_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(curr_file_path, ".."))
from utils import rend_util as ru
from dataio import load_blender
import pickle

# TODO: Refactor code to consolidate the train/val split

def generate_data_maps(intersector, triangle_matrix, mesh, pose, intrinsics, H, W):
    """ Given a pose and a mesh, generate the barycentric maps and face indices
    for the mesh.
    Args:
        intersector: A trimesh.ray.ray_triangle.RayMeshIntersector object.
        triangle_matrix: A (n,3,3) matrix of vertex positions, where each row is vertices
            of a triangle.
        mesh: A trimesh.Trimesh object.
        pose: A (4,4) matrix representing the camera pose.
        intrinsics: A (3,3) matrix representing the camera intrinsics.
        H: Height of the image.
        W: Width of the image.
    Returns:
        dict: With the following keys
              vert_idx: A (H,W, 3) matrix of vertex indices.
              bary_img: A (H,W, 3) matrix of barycentric coordinates.
              depth_img: A (H,W) matrix of depth values.
              is_hit_mask: A (H,W) matrix of 0/1 values indicating whether a ray
                    hit the mesh or not.
              rays_img: A (H,W, 3) matrix of ray directions.
    """
    rays_o, rays_d, _ = ru.get_rays(
            torch.from_numpy(pose).unsqueeze(0).cuda(),
            torch.from_numpy(intrinsics).unsqueeze(0).cuda(),
            H,
            W,
        ) # 1 x HW x 3

    tri_idx, idx_ray, locs = intersector.intersects_id(
        rays_o.cpu().numpy().squeeze(0),
        -rays_d.cpu().numpy().squeeze(0),
        multiple_hits=False,
        return_locations=True) # nhits, nhits, nhits x 3
    
    # -------------------------------------
    # Vertex idx image
    # -------------------------------------
    idxes = np.ones((H*W)) * -1 # WARNING: +1 added later
    idxes[idx_ray] = tri_idx

    idxes = idxes.reshape(H, W)
    idxes = idxes[:, ::-1] # horizontal flip because of our coordinate systems.
    idxes = idxes.astype(np.int)

    # convert tri_idx to vert_idx
    vert_idxer = np.array(mesh.faces) # Nfaces x 3
    pad_idxer = np.array([-1, -1, -1]).reshape(1,3) # will put -1 where there are empty triangles  
    vert_idxer = np.concatenate([pad_idxer, vert_idxer], 0)
    vert_idx = vert_idxer[idxes+1] # H x W x 3 # WARNING: +1 added here

    # -------------------------------------
    # barycentric image
    # -------------------------------------
    hit_tris = triangle_matrix[tri_idx, ...]
    barys = trim.triangles.points_to_barycentric(
        triangles=hit_tris,
        points=locs)

    bary_img = np.zeros((H*W, 3))
    bary_img[idx_ray, :] = barys
    bary_img = bary_img.reshape((H,W,3))
    bary_img = (bary_img[:, ::-1, :]) # horizontal flip because of our coordinate systems.

    # -------------------------------------
    # Hit image
    # -------------------------------------
    is_hit_mask  = intersector.intersects_any(
        rays_o.cpu().numpy().squeeze(0),
        -rays_d.cpu().numpy().squeeze(0),
    )

    is_hit_mask = np.reshape(is_hit_mask, (H,W))
    is_hit_mask = is_hit_mask[:, ::-1]
    
    # -------------------------------------
    # Depths
    # -------------------------------------
    depth_img = np.zeros((H*W))
    depth_flats = locs - rays_o.cpu().numpy()[0,0] # WARNING: broadcasting #HxWx3 -> HxWx3
    depth_img[idx_ray] = np.linalg.norm(depth_flats, ord=2, axis=-1) # N_rays
    depth_img = np.reshape(depth_img, (H,W))
    depth_img = depth_img[:, ::-1]

    # -------------------------------------
    # Ray directions
    # -------------------------------------
    view_dir = -rays_d.cpu().squeeze().numpy()
    view_dir = view_dir.reshape((H,W,3))
    view_dir = view_dir[:, ::-1, :] # horizontal flip because of our coordinate systems.

    return {
        "vert_idx": vert_idx,
        "bary_img": bary_img,
        "is_hit_mask": is_hit_mask,
        "depth_img": depth_img,
        "view_dir": view_dir,
    }



def create_triangle_matrix_from_faces(vertices, faces):
    """ Create a (n,3,3) matrix of vertex positions, where each row is points
    of a triangle.
    """
    triangles = vertices[faces, :]
    return triangles

@torch.no_grad()
def main(args):
    # first load the mesh
    file_ext = os.path.splitext(args.mesh_path)[-1]
    if file_ext == '.obj':
        with open(args.mesh_path, 'r') as fd:
            mesh_dict = trim.exchange.obj.load_obj(fd, maintain_order=True, skip_materials=True)
            # print(mesh_dict.keys())
            if args.offset:
                mesh_dict['vertices'] = mesh_dict['vertices'] + args.offset_value * mesh_dict['vertex_normals']
            mesh = trim.Trimesh(**mesh_dict)
    elif file_ext == '.ply':
        with open(args.mesh_path, 'rb') as fd:
            mesh_dict = trim.exchange.ply.load_ply(fd, fix_texture=False)
            mesh = trim.Trimesh(**mesh_dict)
        
    # mesh = trim.load(args.mesh_path, process=False)
    n_verts = mesh.vertices.shape[0]
    print('Mesh details:')
    print(mesh)

    # creating intersector
    print(f'Creating intersector...')
    intersector = trim.ray.ray_pyembree.RayMeshIntersector(mesh)

    triangle_matrix = create_triangle_matrix_from_faces(mesh.vertices, mesh.faces)

    # Load
    print('Loading scenes...')
    imgs, poses, _, camera_params, i_split, _ = load_blender.load_blender_data(args.indir,
                                                            half_res=False,
                                                            testskip=1,
                                                            splits=['train', 'test'])
    print(f'Loaded {len(i_split[0])} training scenes and {len(i_split[1])} test scenes.')


    H, W, focal = camera_params
    intrinsics = np.array([
                    [focal, 0, 0.5*W, 0],
                    [0, focal, 0.5*H, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]).astype(np.float32)
    
    # training set first
    for i in tqdm(i_split[0]):
        # Now generate the rays for this image using the view matrixs in poses
        curr_img = imgs[i]
        curr_pose = poses[i]
        
        data_dict = generate_data_maps(intersector=intersector,
                                       triangle_matrix=triangle_matrix,
                                       H=H, W=W,
                                       intrinsics=intrinsics,
                                       pose=curr_pose,
                                       mesh=mesh)

        torch.save({'vert_idx_img':data_dict['vert_idx'],
                    'bary_img':data_dict['bary_img'],
                    'is_hit_mask':data_dict['is_hit_mask'],
                    'img': curr_img,
                    'n_verts': n_verts,
                    'view_dir':data_dict['view_dir'],
                    'depth_img':data_dict['depth_img'],},
                os.path.join(args.train_folder, f'r_{i}.pth'))

    for i in tqdm(i_split[1]):
        # Now generate the rays for this image using the view matrixs in poses
        curr_img = imgs[i]
        curr_pose = poses[i]
        
        data_dict = generate_data_maps(intersector=intersector,
                                       triangle_matrix=triangle_matrix,
                                       H=H, W=W,
                                       intrinsics=intrinsics,
                                       pose=curr_pose,
                                       mesh=mesh)

        torch.save({'vert_idx_img':data_dict['vert_idx'],
                    'bary_img':data_dict['bary_img'],
                    'is_hit_mask':data_dict['is_hit_mask'],
                    'img': curr_img,
                    'n_verts': n_verts,
                    'view_dir':data_dict['view_dir'],
                    'depth_img':data_dict['depth_img'],},
                os.path.join(args.test_folder, f'r_{i}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a nerf dataset into pre-rasterized maps')
    parser.add_argument('--indir', '-i', type=str,
     help='Path to the folder contatining the json for scene')
    parser.add_argument('--outdir', '-o', type=str,
        default='',
        help='path to the folder to write maps to')
    parser.add_argument('--mesh_path', '-m', type=str, help='path to the mesh.ply')
    parser.add_argument('--width', '-w', type=int,
        default=800,
        help='width of rendered image')
    parser.add_argument('--height', '-he', type=int,
        default=800,
        help='height of rendered image')
    parser.add_argument('--offset_value', type=float,
        default=0.0,
        help='offset of mesh')
    parser.add_argument('--offset', action='store_true',
        help='offset of mesh')


    args, unkown = parser.parse_known_args()

    # ----------------------------------------------------------
    # Parse args and print the known and unknowns
    # ----------------------------------------------------------
    print('-'*20)
    print('Known args:')
    print(json.dumps(vars(args), indent=2))

    print('-'*20)
    print('Unknown args:')
    if unkown:
        print(json.dumps(vars(unkown), indent=2))

    # -----------------------------------------------------------
    # making some dirs
    # -----------------------------------------------------------
    if not args.outdir:
        args.outdir = os.path.join(args.indir, 'raster')

    os.makedirs(args.outdir, exist_ok=True)

    train_folder = os.path.join(args.outdir, 'train')
    os.makedirs(train_folder, exist_ok=True)
    test_folder = os.path.join(args.outdir, 'test')
    os.makedirs(test_folder, exist_ok=True)

    args.train_folder = train_folder
    args.test_folder = test_folder

    # -----------------------------------------------------------
    # RUN
    # -----------------------------------------------------------
    main(args)
    


########################################################################################################################
# old code

# rays_o, rays_d, _ = ru.get_rays(
#                 torch.from_numpy(curr_pose).unsqueeze(0).cuda(),
#                 torch.from_numpy(intrinsics).unsqueeze(0).cuda(),
#                 H,
#                 W,
#             ) # 1 x HW x 3

#         tri_idx, idx_ray, locs = intersector.intersects_id(
#             rays_o.cpu().numpy().squeeze(0),
#             -rays_d.cpu().numpy().squeeze(0),
#             multiple_hits=False,
#             return_locations=True) # nhits, nhits, nhits x 3
        
#         # from utils.io_util import rays_to_pcd
#         # import open3d as o3d
#         # pcd = rays_to_pcd(rays_o[:, 0].cpu().numpy(), normals=None)
#         # o3d.io.write_point_cloud(f'origin.ply', pcd)

#         # pcd = rays_to_pcd(locs, normals=-rays_d[0, idx_ray, :].cpu().contiguous().numpy())
#         # o3d.io.write_point_cloud(f'hits.ply', pcd)

#         # scaler = torch.arange(0, rays_o.shape[1], device=rays_o.device).unsqueeze(1).repeat(1, 3)
#         # scaler = scaler / rays_o.shape[1]
#         # pcd = rays_to_pcd(rays_o - 4*scaler*rays_d, normals=None)
#         # o3d.io.write_point_cloud(f'all_pixels.ply', pcd)


#         # pcd = rays_to_pcd(rays_o - 4*rays_d, normals=None)
#         # o3d.io.write_point_cloud(f'unif_pixels.ply', pcd)

#         # depths = torch.norm(torch.from_numpy(locs) - rays_o[:, idx_ray, :].cpu().contiguous(), dim=2)
#         # depths_img = np.zeros((H*W))
#         # depths_img[idx_ray] = depths
#         # depths_img = depths_img.reshape(W, H)[:, ::-1]
#         # import matplotlib.pyplot as plt
#         # plt.imshow(depths_img)
#         # plt.savefig('aa.png')

#         # depths_ = depths.unsqueeze(-1).repeat(1, 1, 3)
#         # computed_hits = rays_o[:, idx_ray, :] - depths_.cuda() * rays_d[:, idx_ray, :]
#         # computed_hits = computed_hits.cpu().numpy()
#         # pcd = rays_to_pcd(computed_hits, normals=None)
#         # o3d.io.write_point_cloud(f'computed_hits.ply', pcd)



#         # sys.exit()

#         # -------------------------------------
#         # Vertex idx image
#         # -------------------------------------
#         idxes = np.ones((H*W)) * -1
#         idxes[idx_ray] = tri_idx

#         idxes = idxes.reshape(H, W)
#         idxes = idxes[:, ::-1] # horizontal flip because of our coordinate systems.
#         idxes = idxes.astype(np.int)

#         # import matplotlib.pyplot as plt
#         # plt.imshow(idxes == 5); plt.colorbar()
#         # plt.savefig('idx5.png')
#         # plt.close()
#         # plt.imshow(idxes == 389); plt.colorbar()
#         # plt.savefig('idx389.png')
#         # sys.exit()

#         # convert tri_idx to vert_idx
#         vert_idxer = np.array(mesh.faces) # Nfaces x 3
#         print(np.min(vert_idxer), np.max(vert_idxer))
#         pad_idxer = np.array([-1, -1, -1]).reshape(1,3) # will put -1 where there are empty triangles  
#         vert_idxer = np.concatenate([pad_idxer, vert_idxer], 0)

#         vert_idx = vert_idxer[idxes+1] # H x W x 3

#         # import matplotlib.pyplot as plt
#         # plt.imshow(vert_idx[..., 0] == -1)
#         # plt.savefig('vert_idx_0.png')
#         # plt.close()
#         # plt.imshow(vert_idx[..., 0] == 160)
#         # plt.savefig('vert_idx_1.png')

#         # sys.exit()
#         # -------------------------------------
#         # barycentric image
#         # -------------------------------------
#         hit_tris = triangle_matrix[tri_idx, ...]
#         barys = trim.triangles.points_to_barycentric(
#             triangles=hit_tris,
#             points=locs)

#         bary_img = np.zeros((H*W, 3))
#         bary_img[idx_ray, :] = barys
#         bary_img = bary_img.reshape((H,W,3))
#         bary_img = (bary_img[:, ::-1, :]) # horizontal flip because of our coordinate systems.
        
#         # import matplotlib.pyplot as plt
#         # plt.imshow(bary_img[..., 0])
#         # plt.savefig('bary_0.png')
#         # plt.close()
#         # plt.imshow(bary_img[..., 1])
#         # plt.savefig('bary_1.png')
#         # plt.close()
#         # plt.imshow(bary_img[..., 2])
#         # plt.savefig('bary_2.png')
#         # plt.close()
#         # plt.imshow(np.sum(bary_img, axis=-1))
#         # plt.savefig('bary_sum.png')
#         # plt.close()

#         # closest = trim.triangles.closest_point(triangles=hit_tris, points=locs)
#         # print('diff', np.max(np.abs(closest - locs)))


#         # sys.exit()

#         # -------------------------------------
#         # Hit image
#         # -------------------------------------
#         is_hit_mask  = intersector.intersects_any(
#             rays_o.cpu().numpy().squeeze(0),
#             -rays_d.cpu().numpy().squeeze(0),
#         )

#         is_hit_mask = np.reshape(is_hit_mask, (H,W))
#         is_hit_mask = is_hit_mask[:, ::-1]
        
#         # -------------------------------------
#         # Depths
#         # -------------------------------------
#         depth_img = np.zeros((H*W))
#         depth_flats = locs - rays_o.cpu().numpy()[0,0] # WARNING: broadcasting #HxWx3 -> HxWx3
#         depth_img[idx_ray] = np.linalg.norm(depth_flats, ord=2, axis=-1) # N_rays
#         depth_img = np.reshape(depth_img, (H,W))
#         depth_img = depth_img[:, ::-1]

#         # -------------------------------------
#         # Ray directions
#         # -------------------------------------
#         rays_img = -rays_d.cpu().squeeze().numpy()
#         rays_img = rays_img.reshape((H,W,3))
#         rays_img = rays_img[:, ::-1, :] # horizontal flip because of our coordinate systems.

#         if i == 1:
#             print(np.min(rays_img), np.max(rays_img))

#             import matplotlib.pyplot as plt
#             def rescale(img):
#                 return (img + 1) / 2
#             plt.imshow(rescale(rays_img))
#             plt.savefig('vdir.png')
#             plt.close()
#             plt.imshow(rescale(rays_img)[..., 0])
#             plt.savefig('vdir_1.png')
#             plt.close()
#             plt.imshow(rescale(rays_img)[..., 1])
#             plt.savefig('vdir_2.png')
#             plt.close()
#             plt.imshow(rescale(rays_img)[..., 2])
#             plt.savefig('vdir_3.png')
#             plt.close()

#             sys.exit()