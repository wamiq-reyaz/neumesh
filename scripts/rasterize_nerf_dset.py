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


# TODO: load a mesh
# generate a bunch of rays from view points
# perform intersection
# find barycentric coordinates of the intersection
# save those to a file


def create_triangle_matrix_from_faces(vertices, faces):
    """ Create a (n,3,3) matrix of vertex positions, where each row is points
    of a triangle.
    """
    triangles = vertices[faces, :]
    return triangles


# def process_and_save()

@torch.no_grad()
def main(args):
    # first load the mesh
    mesh = trim.load(args.mesh_path, process=False)
    n_verts = mesh.vertices.shape[0]
    print('Mesh details:')
    print(mesh)

    # creating intersector
    print(f'Creating intersector...')
    intersector = trim.ray.ray_pyembree.RayMeshIntersector(mesh)

    triangle_matrix = create_triangle_matrix_from_faces(mesh.vertices, mesh.faces)

    # Load
    print('Loading scenes...')
    imgs, poses, render_poses, camera_params, i_split, fnames = load_blender.load_blender_data(args.indir,
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
        
        rays_o, rays_d, _ = ru.get_rays(
                torch.from_numpy(curr_pose).unsqueeze(0).cuda(),
                torch.from_numpy(intrinsics).unsqueeze(0).cuda(),
                H,
                W,
            )

        tri_idx, idx_ray, locs = intersector.intersects_id(
            rays_o.cpu().numpy().squeeze(0),
            -rays_d.cpu().numpy().squeeze(0),
            multiple_hits=False,
            return_locations=True)
        
        idxes = np.zeros((H*W))
        idxes[idx_ray] = tri_idx

        idxes = idxes.reshape(H, W)
        idxes = idxes[:, ::-1] # horizontal flip because of our coordinate systems.
        idxes = idxes.astype(np.int)

        # convert tri_idx to vert_idx

        vert_idxer = np.array(mesh.faces) # Nfaces x 3
        pad_idxer = np.array([-1, -1, -1]).reshape(1,3) # will put -1 where there are empty triangles  
        vert_idxer = np.concatenate([pad_idxer, vert_idxer], 0)

        vert_idx = vert_idxer[idxes] # H x W x 3

        hit_tris = triangle_matrix[tri_idx, ...]
        barys = trim.triangles.points_to_barycentric(
            triangles=hit_tris,
            points=locs)

        bary_img = np.zeros((H*W, 3))
        bary_img[idx_ray, :] = barys
        bary_img = bary_img.reshape((H,W,3))
        bary_img = (bary_img[:, ::-1, :]) # horizontal flip because of our coordinate systems.
        
        
        is_hit_mask  = intersector.intersects_any(
            rays_o.cpu().numpy().squeeze(0),
            -rays_d.cpu().numpy().squeeze(0),
        )

        is_hit_mask = np.reshape(is_hit_mask, (H,W))
        is_hit_mask = is_hit_mask[:, ::-1]


        torch.save({'vert_idx_img':vert_idx,
                    'bary_img':bary_img,
                    'is_hit_mask':is_hit_mask,
                    'img': curr_img},
                os.path.join(args.train_folder, f'r_{i}.pth'))

    for i in tqdm(i_split[1]):
        # Now generate the rays for this image using the view matrixs in poses
        curr_img = imgs[i]
        curr_pose = poses[i]
        
        rays_o, rays_d, _ = ru.get_rays(
                torch.from_numpy(curr_pose).unsqueeze(0).cuda(),
                torch.from_numpy(intrinsics).unsqueeze(0).cuda(),
                H,
                W,
            )

        tri_idx, idx_ray, locs = intersector.intersects_id(
            rays_o.cpu().numpy().squeeze(0),
            -rays_d.cpu().numpy().squeeze(0),
            multiple_hits=False,
            return_locations=True)
        
        idxes = np.zeros((H*W))
        idxes[idx_ray] = tri_idx

        idxes = idxes.reshape(H, W)
        idxes = idxes[:, ::-1] # horizontal flip because of our coordinate systems.
        idxes = idxes.astype(np.int)

        # convert tri_idx to vert_idx

        vert_idxer = np.array(mesh.faces) # Nfaces x 3
        pad_idxer = np.array([-1, -1, -1]).reshape(1,3) # will put -1 where there are empty triangles  
        vert_idxer = np.concatenate([pad_idxer, vert_idxer], 0)

        vert_idx = vert_idxer[idxes] # H x W x 3

        hit_tris = triangle_matrix[tri_idx, ...]
        barys = trim.triangles.points_to_barycentric(
            triangles=hit_tris,
            points=locs)

        bary_img = np.zeros((H*W, 3))
        bary_img[idx_ray, :] = barys
        bary_img = bary_img.reshape((H,W,3))
        bary_img = (bary_img[:, ::-1, :]) # horizontal flip because of our coordinate systems.
        
        
        is_hit_mask  = intersector.intersects_any(
            rays_o.cpu().numpy().squeeze(0),
            -rays_d.cpu().numpy().squeeze(0),
        )

        is_hit_mask = np.reshape(is_hit_mask, (H,W))
        is_hit_mask = is_hit_mask[:, ::-1]


        torch.save({'vert_idx_img':vert_idx,
                    'bary_img':bary_img,
                    'is_hit_mask':is_hit_mask,
                    'img': curr_img,
                    'n_verts': n_verts},
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
    