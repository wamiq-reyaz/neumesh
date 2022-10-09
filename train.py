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
import torchvision.transforms as T



from utils import io_util, rend_util
from utils.checkpoints import sorted_ckpts
from utils.dist_util import set_seed

from dataio import get_data, load_blender

from models.frameworks.neumesh import get_neumesh_model

from render import create_render_args


import wandb
import json


PROJECT = 'editable_nerf'
HACKY_ITER_COUNT = 0

def integerify(img):
    return (img * 255.0).astype(np.uint8)

def integerify(img):
    return (img * 255.0).astype(np.uint8)

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


        if config['invert_z']:
            rays_o = torch.matmul(torch.tensor([[1, 0, 0.],
                                            [0, 1, 0.],
                                            [0, 0, -1]], device=rays_o.device).unsqueeze(0),
                                rays_o[..., None]).squeeze(-1) # (1x3x3) x (nx3x1) and squeeze

            # rotate to invert z on the directions
            rays_d = torch.matmul(torch.tensor([[1, 0, 0.],
                                            [0, 1, 0.],
                                            [0, 0, -1]], device=rays_d.device).unsqueeze(0),
                                rays_d[..., None]).squeeze(-1) # (1x3x3) x (nx3x1) and squeeze


        # sample GT RGB values
        selected_idx = selected_idx.squeeze(0) # N
        curr_img = curr_img[..., :3] # HxWx3 drop alpha

        if config['invert_z'] and config.get('invert_z_gt', False):
            curr_img = curr_img[:, ::-1, :] # left-right flip

        curr_img = curr_img.flip((1,)) # left-right flip

        curr_img = curr_img.reshape(-1, 3) # HxWx3 -> (HxW)x3

        gt_rgb = curr_img[selected_idx] # Nx3

        if curr_iter == 0:
            print(f'render_kwargs_test: {render_kwargs_test}')
        rgb, depth, extras = render_fn(rays_o,
                        -rays_d,
                        show_progress=False,
                        detailed_output=True,
                        **render_kwargs_test)
        

        loss = F.mse_loss(rgb.squeeze(), gt_rgb.float().squeeze())
        sdf = extras['implicit_nablas']
        sdf_norm = torch.norm(sdf, dim=-1)
        eikonal_loss = F.mse_loss(sdf_norm.squeeze(), torch.ones_like(sdf_norm).squeeze())

        loss = loss + config['neumesh']['loss_weights']['eikonal'] * eikonal_loss

        # eikonal loss

        optimizer.zero_grad()
        warmup_steps = config['training']['scheduler'].get('warmup_steps', 0)
        if (curr_iter < warmup_steps) and (warmup_steps > 0):
            # warmup learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = float(curr_iter+1)/warmup_steps * config['training']['lr']
                wandb.log({'train/lr': param_group['lr']}, step=curr_iter)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['training']['lr'] * (0.999999**curr_iter)
                wandb.log({'train/lr': param_group['lr']}, step=curr_iter)

        wandb.log({'train/eikonal_loss': eikonal_loss.item()}, step=curr_iter)

        loss.backward()
        optimizer.step()

        return loss.clone().detach().cpu().numpy()
        

def val(config, curr_iter, model, optimizer, imgs, poses, render_poses, camera_params, i_split, render_kwargs_test, render_fn):
    return None, None

def render_val(config, model, render_kwargs_test, render_fn, pose, camera_params):
    """ reimplementation of render_function in render.py
    """

    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose)
    # construct intrinsic matrix
    H, W, focal = camera_params
    # print(f'H, W, focal: {H, W, focal}')
    # print('type of H, W, focal: ', type(H), type(W), type(focal))
    # scale by the validation scale
    H /= config['data']['val_downscale']
    W /= config['data']['val_downscale']
    focal /= config['data']['val_downscale']

    H, W, focal = int(H), int(W), focal

    intrinsics = np.array([
        [focal, 0, 0.5*W, 0],
        [0, focal, 0.5*H, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ]).astype(np.float32)

    rays_o, rays_d, _ = rend_util.get_rays(
            pose.float().unsqueeze(0).cuda(),
            torch.from_numpy(intrinsics).unsqueeze(0),
            H,
            W,
            N_rays=-1
        )

    if config['invert_z']:
        rays_o = torch.matmul(torch.tensor([[1, 0, 0.],
                                            [0, 1, 0.],
                                            [0, 0, -1]], device=rays_o.device).unsqueeze(0),
                                rays_o[..., None]).squeeze(-1) # (1x3x3) x (nx3x1) and squeeze

        # rotate to invert z on the directions
        rays_d = torch.matmul(torch.tensor([[1, 0, 0.],
                                            [0, 1, 0.],
                                            [0, 0, -1]], device=rays_d.device).unsqueeze(0),
                                rays_d[..., None]).squeeze(-1) # (1x3x3) x (nx3x1) and squeeze

    global HACKY_ITER_COUNT
    if args.debug and (HACKY_ITER_COUNT == 5):
        print('saving rays as a point cloud with normals')
        points = torch.cat([rays_o-rays_d, rays_o - 4*rays_d], dim=1)
        pcd = io_util.rays_to_pcd(points=points, normals=rays_d)
        o3d.io.write_point_cloud(f'rays_tex_2{HACKY_ITER_COUNT:04d}.ply', pcd)
        print(f'Saved rays to rays.ply')
    HACKY_ITER_COUNT += 1

    with torch.no_grad():   
        rgb, _, extras = render_fn(
            rays_o,
            -rays_d,
            show_progress=False,
            detailed_output=False,
            **render_kwargs_test
        )

        rgb = rgb.reshape(H, W, 3).unsqueeze(0)
        img = integerify(rgb.cpu().numpy())

        mask_volume = extras['mask_volume'].cpu()
        mask_volume = mask_volume.reshape(H, W).unsqueeze(0).numpy()

        images = wandb.Image(img, caption="Validation Image")
        
        wandb.log({"val/images": images})

    return img, mask_volume



if __name__ == '__main__':
    parser = io_util.create_args_parser()
    parser = create_render_args(parser)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--half_res', action='store_true')
    parser.add_argument('--just_render', action='store_true')
    parser.add_argument('--render_path', type=str, default='renders')
    parser.add_argument('--tags', type=str, default='')
    parser.add_argument('--invert_z', action='store_true')
    parser.add_argument('--notes', type=str, default='')
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
    imgs, poses, render_poses, camera_params, i_split, fnames = load_blender.load_blender_data(config['data']['data_dir'],
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

    print(f'fnames[0] = {fnames[0]} and {i_split[0][0]} and {fnames[i_split[0][0]]}')

    # ------------------------------------------------------------------------------
    # setup experiments directory
    # ------------------------------------------------------------------------------
    expt_prefix = io_util.gen_expt_prefix(debug=args.debug, render=args.just_render)
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
    if args.tags:
        tags = args.tags.split(',')
    else:
        tags = None
    curr_run = wandb.init(project=PROJECT,
                          config=config,
                          name=expt_prefix,
                          save_code=True,
                          tags=tags,
                          notes=args.notes)
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

        if args.just_render:
            os.makedirs(args.render_path, exist_ok=True)
            for ii in i_split[1]:
                img, mask_volume = render_val(config, model, render_kwargs_test, render_fn, poses[ii], camera_params)
                # save the rgb
                save_path = os.path.join(args.render_path, f'{ii:05d}.png')
                print(f'Saving image to {save_path}')
                imageio.imwrite(save_path, np.squeeze(img))

                # save the mask volume
                save_path = os.path.join(args.render_path, f'volume_{ii:05d}.png')
                print(f'Saving volume to {save_path}')
                imageio.imwrite(save_path, np.squeeze(mask_volume))


            wandb.finish()
            import sys
            sys.exit()

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
            pose_to_render = random.choice(i_split[1])
            render_pose = torch.from_numpy(poses[pose_to_render])
            render_val(config, model, render_kwargs_test, render_fn, render_pose, camera_params)

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
