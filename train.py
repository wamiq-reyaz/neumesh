import os

import imageio
import numpy as np
from tqdm import tqdm, trange
import open3d as o3d
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


from utils import io_util, rend_util
from utils.checkpoints import sorted_ckpts
from utils.dist_util import set_seed

from dataio import get_data, load_blender

from models.frameworks.neumesh import get_neumesh_model

from render import create_render_args


import wandb
import json


PROJECT = 'editable_nerf'


def main(args):
    pass

def train(config, curr_iter, model, optimizer, imgs, poses, render_poses, camera_params, i_split, render_kwargs_test, render_fn):
    # create intrinsics
    H, W, focal = camera_params
    intrinsics = np.array([
        [focal, 0, 0.5*W, 0],
        [0, focal, 0.5*H, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ]).astype(np.float32)

    if config['training'].get('sampling', 'rays_per_image') == 'rays_per_image':
        # choose an image and sample points and rays on that image
        i_img = random.choice(i_split[0]) 
        curr_img = imgs[i_img] # HxWx4
        curr_pose = poses[i_img] # 4x4
        curr_img = torch.from_numpy(curr_img).cuda() # HxWx4

        # generate all rays
        rays_o, rays_d, selected_idx = rend_util.get_rays(
            torch.from_numpy(curr_pose).unsqueeze(0).cuda(),
            torch.from_numpy(intrinsics).unsqueeze(0).cuda(),
            H,
            W,
            config['data']['N_rays']
        )

        # rays_o: 1xNx3, rays_d: 1xNx3, selected_idx: 1xN
        # curr_img: HxWx4

        # print(f'rays_o: {rays_o.shape}, rays_d: {rays_d.shape}, selected_idx: {selected_idx.shape}')
        # print(f'devices: rays_o: {rays_o.device}, rays_d: {rays_d.device}, selected_idx: {selected_idx.device}')
        # print(f'image device: {curr_img.device}')
        # sample GT RGB values
        selected_idx = selected_idx.squeeze(0) # N
        curr_img = curr_img[..., :3] # HxWx3 drop alpha
        curr_img = curr_img.reshape(-1, 3) # HxWx3 -> (HxW)x3
        gt_rgb = curr_img[selected_idx] # Nx3

        if curr_iter == 0:
            print(f'render_kwargs_test: {render_kwargs_test}')
        rgb, _, _ = render_fn(rays_o,
                        -rays_d,
                        show_progress=False,
                        detailed_output=False,
                        **render_kwargs_test)
        

        loss = F.mse_loss(rgb, gt_rgb.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.clone().detach().cpu().numpy()
        

def val(config, curr_iter, model, optimizer, imgs, poses, render_poses, camera_params, i_split, render_kwargs_test, render_fn):
    return None, None

def render_val(config, model, render_kwargs_test, render_fn, pose, camera_params):
    """ reimplementation of render_function in render.py
    """
    def integerify(img):
        return (img * 255.0).astype(np.uint8)

    # construct intrinsic matrix
    H, W, focal = camera_params
    # print(f'H, W, focal: {H, W, focal}')
    # print('type of H, W, focal: ', type(H), type(W), type(focal))
    # scale by the validation scale
    H /= config['data']['val_downscale']
    W /= config['data']['val_downscale']
    focal /= config['data']['val_downscale']

    H, W, focal = int(H), int(W), int(focal)

    intrinsics = np.array([
        [focal, 0, 0.5*W, 0],
        [0, focal, 0.5*H, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ]).astype(np.float32)

    # intrinsics = np.array(
    #    [[ 2.8923e+03, -2.6210e-04,  8.2321e+02,  0.0000e+00],
    #     [ 0.0000e+00,  2.8832e+03,  6.1907e+02,  0.0000e+00],
    #     [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00],
    #     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
    # ).astype(np.float32)

    # pose = torch.tensor([[ 0.23530716,  0.33113482, -0.91377255,  4.78497434],
    #         [-0.77687748,  0.62903356,  0.02789544 , 0.09405055],
    #         [ 0.58403075,  0.70332532 , 0.40526728 ,-2.25775732],
    #         [ 0. ,         0.      ,    0.  ,        1.        ]])

    rays_o, rays_d, _ = rend_util.get_rays(
            pose.float().unsqueeze(0).cuda(),
            torch.from_numpy(intrinsics).unsqueeze(0),
            H,
            W,
            N_rays=-1
        )

    # print(f'type of rays_o, rays_d: {type(rays_o), type(rays_d)}')
    # print(f'rays_o.shape: {rays_o.shape}')
    # print(f'rays_d.shape: {rays_d.shape}')
    if args.debug:
        print('saving rays as a point cloud with normals')
        # furhter = rays_d * 10
        points = torch.cat([rays_o+rays_d, rays_o + 4*rays_d], dim=1)
        pcd = io_util.rays_to_pcd(points=points, normals=rays_d)
        o3d.io.write_point_cloud('rays.ply', pcd)
        print(rays_o)

    with torch.no_grad():   
        rgb, _, _ = render_fn(
            rays_o,
            -rays_d,
            show_progress=False,
            detailed_output=False,
            **render_kwargs_test
        )
        print('=' * 10)
        print('printing rendered info')
        print(type(rgb))
        print(rgb.shape)
        print(torch.unique(rgb))
        rgb = rgb.reshape(H, W, 3).unsqueeze(0)

        img = integerify(rgb.cpu().numpy())



        images = wandb.Image(img, caption="Validation Image")
        
        print(type(images))
        print(dir(images))
          
        wandb.log({"val/images": images})



if __name__ == '__main__':
    parser = io_util.create_args_parser()
    parser = create_render_args(parser)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--half_res', action='store_true')
    args, unknown = parser.parse_known_args()

    config = io_util.load_config(args, {'exp_name': 'test'})
    # print(f'Loading config from {args.config}')
    print(f'This is the config: \n {json.dumps(config, indent=4)}')

    # ------------------------------------------------------------------------------
    # set the seeds
    # ------------------------------------------------------------------------------
    set_seed(args.seed)
    

    # ------------------------------------------------------------------------------
    # load the model from the config
    # ------------------------------------------------------------------------------

    (model, render_kwargs_test, render_fn) = get_neumesh_model(config)
    
    if args.load_pt is not None:
        print(f'Loading model from {args.load_pt}')
        model.load_state_dict(torch.load(args.load_pt, map_location='cuda:0')['model'])
        print('Loaded model')

    model = model.cuda()


    print(f'Loaded model: {model}')
    print(f'Loaded render_kwargs_test: {json.dumps(render_kwargs_test, indent=4)}')

    # ------------------------------------------------------------------------------
    # load the dataset
    # ------------------------------------------------------------------------------
    print(f'Loading dataset...')
    imgs, poses, render_poses, camera_params, i_split = load_blender.load_blender_data(config['data']['data_dir'],
                                                            half_res=args.half_res,
                                                            testskip=1,
                                                            splits=['train', 'test'])
    
    print(f'Loaded dataset: \n\
            Images: {imgs.shape},\n\
            Poses: {poses.shape}, \n\
            Render Poses; {render_poses.shape}, \n\
            Camera Intrinsics: {camera_params},\n\
            Train images: {len(i_split[0])}, \n\
            Test images: {len(i_split[1])}')

    # ------------------------------------------------------------------------------
    # setup experiments directory
    # ------------------------------------------------------------------------------
    expt_prefix = io_util.gen_expt_prefix(debug=args.debug)
    full_expt_path = os.path.join('.', 'out', config['expname'], expt_prefix)
    print(f'Saving experiment to {full_expt_path}')
    print(f'Current experiment is {expt_prefix}')
    os.makedirs(full_expt_path, exist_ok=True)
    os.makedirs(os.path.join(full_expt_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(full_expt_path, 'code'), exist_ok=True)

    print(f'Copying code to {os.path.join(full_expt_path, "code")}')
    io_util.backup(os.path.join(full_expt_path, "code"))

    # ------------------------------------------------------------------------------
    # setup wandb
    # ------------------------------------------------------------------------------
    curr_run = wandb.init(project=PROJECT, config=config, name=expt_prefix)
    wandb.watch(model)


    # ------------------------------------------------------------------------------
    # setup optimizer
    # ------------------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])


    # ------------------------------------------------------------------------------
    # main training loop
    # ------------------------------------------------------------------------------
    for curr_iter in trange(config['training']['num_iters']):
        # TODO: implement warmup

        curr_loss = train(config, curr_iter, model, optimizer, imgs, poses, render_poses, camera_params, i_split, render_kwargs_test, render_fn)
        wandb.log({"train/loss": curr_loss}, step=curr_iter)
        
        if curr_iter % config['training']['i_val'] == 0:
            psnr, mse = val(config, curr_iter, model, optimizer, imgs, poses, render_poses, camera_params, i_split, render_kwargs_test, render_fn)
        
        # ------------------------------------------------------------------------------
        # save checkpoints
        # ------------------------------------------------------------------------------
        if curr_iter % config['training']['i_save'] == 0:
            io_util.save_ckpt(model,
                        os.path.join(full_expt_path, 'checkpoints'),
                        curr_iter,
                        optimizer=optimizer,
                        scheduler=None,
                        pnsr=psnr,
                        mse=mse) # TODO add scheduler

        

        
       
        # ------------------------------------------------------------------------------
        # save images
        # ------------------------------------------------------------------------------

        if curr_iter % config['training']['i_render'] == 0:
            render_val(config, model, render_kwargs_test, render_fn, render_poses[0], camera_params)

    # ------------------------------------------------------------------------------
    # save final checkpoint
    # ------------------------------------------------------------------------------
    io_util.save_ckpt(model,
                    os.path.join(full_expt_path, 'checkpoints'),
                    curr_iter,
                    optimizer=optimizer,
                    scheduler=None,
                    pnsr=psnr,
                    mse=mse) # TODO add scheduler

    # ------------------------------------------------------------------------------
    # save final images
    # ------------------------------------------------------------------------------
    # render_val(args, model, render_kwargs_test, render_fn, render_poses[0])

    # # ------------------------------------------------------------------------------
    # # save final model
    # # ------------------------------------------------------------------------------
    # io_util.save_model(model, os.path.join(full_expt_path, 'model.pth'))

    # # ------------------------------------------------------------------------------
    # # save final config
    # # ------------------------------------------------------------------------------
    # io_util.save_config(config, os.path.join(full_expt_path, 'config.json'))

    print(f'Finished training {expt_prefix}')

    # ------------------------------------------------------------------------------

    





# if __name__ == '__main__':
#     parser = io_util.create_args_parser()
#     parser = create_render_args(parser)
#     args, unknown = parser.parse_known_args()
#     config = io_util.load_config(args, unknown)

#     pretty_config = json.dumps(config, indent=4)
#     print(pretty_config)


#     ## debug the loader

#     # sheep dataset 
#     imgs, poses, render_poses, camera_params, i_split = load_blender.load_blender_data(config['data']['data_dir'],
#                                                             half_res=False,
#                                                             testskip=1,
#                                                             splits=['train', 'test'])

#     # lego dataset
#     imgs, poses, render_poses, camera_params, i_split = load_blender.load_blender_data('/home/parawr/Projects/instant-ngp/data/nerf/nerf_synthetic/lego',
#                                                         half_res=False,
#                                                         testskip=1,
#                                                         splits=['train', 'test'])


#     # ------------
#     #  do the images look alright
#     # ------------

#     import matplotlib.pyplot as plt
#     plt.imshow(imgs[0])
#     # plt.show()
#     plt.savefig('playground/train/img_lego_train_0.png')

#     # ------------
#     #  can you generate rays from the poses and render them?
#     # ------------

#     # perhaps use load_K_Rt_from_P from utils/rend_util
#     # decompose_blender_mat_to_K_Rt(pose)
#     def decompose_blender_mat_to_intrinsics_pose(pose_4x4):
#         p = pose_4x4[:3, :]
#         intrinsics, pose = rend_util.load_K_Rt_from_P(p)

#         return intrinsics, pose
        
#     intrinsics, pose = decompose_blender_mat_to_intrinsics_pose(poses[0])

#     print('Intrinsics', intrinsics)
#     print('Pose', pose)
#     print('full_mat', poses[0])

#     print('lalalala')


#     # check if pose has valid rotation matrix
#     rot_matrix = pose[:3, :3]
#     print('rot_matrix', rot_matrix)
#     print('det', np.linalg.det(rot_matrix))

#     ## can you generate rays from the poses and render them?

#     # Checks if a matrix is a valid rotation matrix.
#     def isRotationMatrix(R) :
#         Rt = np.transpose(R)
#         shouldBeIdentity = np.dot(Rt, R)
#         I = np.identity(3, dtype = R.dtype)
#         n = np.linalg.norm(I - shouldBeIdentity)
#         return n < 1e-6

#     import math

#     # Calculates rotation matrix to euler angles
#     # The result is the same as MATLAB except the order
#     # of the euler angles ( x and z are swapped ).
#     def rotationMatrixToEulerAngles(R) :

#         assert(isRotationMatrix(R))

#         sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

#         singular = sy < 1e-6

#         if  not singular :
#             x = math.atan2(R[2,1] , R[2,2])
#             y = math.atan2(-R[2,0], sy)
#             z = math.atan2(R[1,0], R[0,0])
#         else :
#             x = math.atan2(-R[1,2], R[1,1])
#             y = math.atan2(-R[2,0], sy)
#             z = 0

#         return np.array([x, y, z])


#     # output the result as euler angle to visually confirm results
#     print('euler angles', rotationMatrixToEulerAngles(rot_matrix.T))
#     print('H W focal', camera_params)