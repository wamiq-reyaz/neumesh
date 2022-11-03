import os
import sys
sys.path.append('..')
import argparse

import numpy as np
import torch
from natsort import natsorted
from glob import glob
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

from models.frameworks import raster_caster
from utils.metric_util import psnr, ssim
from models.frameworks.raster_caster import alpha_composite_rgbs


from moviepy.editor import *
def vid_from_list(inlist, duration):
    clips = [ImageClip(m).set_duration(duration)
              for m in inlist]
    video = concatenate_videoclips(clips, method="compose")
    return video

def create_video(gtl, predl, duration):
    n_frames = len(gt)
    duration_per_frame = duration/float(n_frames)
    gts = vid_from_list(gtl, duration=duration_per_frame)
    preds = vid_from_list(predl, duration=duration_per_frame)
    final_clip = clips_array([[gts, preds]])
    final_clip.write_videofile('first_rgba_deformed.mp4', fps=24)


base_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/data/editable_nerf/neus_sheep/raster_512_scalex_0.8/train/r_'
# base_test_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/data/editable_nerf/neus_sheep/raster_512/test/r_'
# base_test_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/data/editable_nerf/neus_sheep/raster_512_scalex_0.8/test/r_'
base_test_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/data/editable_nerf/neus_sheep/raster_512_scalex_0.8/test/r_'
suffix = '.pth'

# b_in = base_path.replace('deform_11k_offset0', 'deform_11k_offsetm0.03')
# b_in_test = base_test_path.replace('raster_512', 'raster_512_offm0.03')
b_in_test = base_test_path.replace('raster_512_scalex_0.8', 'raster_512_th_n0.03_scalex_0.8')

# b_off = base_path.replace('deform_11k_offset0', 'deform_11k_offset0.03')
# b_off_test = base_test_path.replace('raster_512', 'raster_512_off0.03')
b_off_test = base_test_path.replace('raster_512_scalex_0.8', 'raster_512_th_0.03_scalex_0.8')


# /datawaha/cggroup/parawr/Projects/adobe/neumesh/out/test_ckpt/neus_sheep_shell_500k_2main_1_view_val0.9/
ckpt_base_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/out/test_ckpt/neus_sheep_500k_2main_2_view_val0.9/'
ckpt_base_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/out/test_ckpt/neus_sheep_shell_500k_2main_1_view_val0.9_drop_vertidentity/'
ckpt_base_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/out/test_ckpt/neus_sheep_shell_500k_2main_1_view_val0.9_drop_vertidentity_rgba'
# ckpt_base_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/out/test_ckpt/neus_sheep_shell_500k_2main_1_view_val0.9/'

def sample_to_dict(sample):
    is_hit_mask = sample['is_hit_mask']
    img = sample['img']    
    data_dict = {
    'vert_idx_img': torch.from_numpy(sample['vert_idx_img']).cuda().unsqueeze(0).long(),
    'bary_img': torch.from_numpy(sample['bary_img']).cuda().unsqueeze(0).float(),
    'is_hit_mask': torch.from_numpy(sample['is_hit_mask']).cuda().unsqueeze(0).float(),
    # 'img':torch.from_numpy(img[:, :, :3]).cuda().unsqueeze(0).float(),

    'img':torch.from_numpy(img[:, :, :]).cuda().unsqueeze(0).float(),
    # 'view_dir':  torch.from_numpy(sample['view_dir']).cuda().unsqueeze(0).float(),
    'view_dir':  sample['view_dir'].reshape(1, 800,800,3).cuda(),

    }

    return data_dict


def collate_fn(in_list):
    keys = in_list[0].keys()
    ret_dict = dict()
    for k in keys:
        if k not in [ 'img', 'view_dir']:
            ret_dict[k] = torch.cat([ll[k] for ll in in_list], dim=-1)
        else:
            ret_dict[k] = in_list[0][k]
    return ret_dict

def to_img(tensor):
    return (tensor.cpu().squeeze().numpy()*255).astype(np.uint8)

    
# d1 = [torch.load(base_test_path + str(i) + suffix) for i in range(200,201)]

# test_dataset = [torch.load(base_test_path + str(i) + suffix) for i in range(100)]
# bin_test_dataset = [torch.load(b_in_test + str(i) + suffix) for i in range(100)]
# boff_test_dataset = [torch.load(b_off_test + str(i) + suffix) for i in range(100)]


test_dataset = [torch.load(base_test_path + str(i) + suffix) for i in range(100,300)]
bin_test_dataset = [torch.load(b_in_test + str(i) + suffix) for i in range(100,300)]
boff_test_dataset = [torch.load(b_off_test + str(i) + suffix) for i in range(100,300)]

test_dict = [sample_to_dict(i) for i in test_dataset]
bin_test_dict = [sample_to_dict(i) for i in bin_test_dataset]
boff_test_dict = [sample_to_dict(i) for i in boff_test_dataset]


# n_verts = d1[0]['n_verts']
# # n_verts = test_dataset[0]['n_verts']
# model = raster_caster.RasterCaster(
#     n_verts= n_verts+1,
#     n_chan=3,
#     n_hidden=128,
#     depth=2,
#     use_viewdirs=True,
#     viewdirs_depth=1,
#     viewdirs_attach=0,
#     n_mesh=1
# )


n_verts = 570000
model = raster_caster.RasterCaster(
    n_verts= n_verts+1,
    n_chan=4,
    n_hidden=64,
    depth=1,
    use_viewdirs=True,
    viewdirs_depth=1,
    viewdirs_attach=0,
    n_mesh=3
)



ckpts = natsorted(glob(os.path.join(ckpt_base_path, '*')))


overall_psnr = []
overall_ssim = []

range_tuple = (15,16)
for ii in trange(*range_tuple):
# for ii in trange(len(ckpts)):
    curr_ckpt = torch.load(ckpts[ii], map_location='cpu')
    model.load_state_dict(curr_ckpt)
    model = model.cuda()
    model.eval()
    
    preds = []
    gt = []
    with torch.no_grad():
        for idx, _  in  tqdm(enumerate(test_dict), total=len(test_dict)):
            data_dict = collate_fn([test_dict[idx], bin_test_dict[idx], boff_test_dict[idx]])
            pred = model(data_dict)
            rgb, alpha = alpha_composite_rgbs(pred)

            # out = torch.sigmoid(model(data_dict))
            rgb = rgb.clamp(0,1)    
            preds.append(to_img(rgb))
            gt.append(to_img(data_dict['img']))

            
        # psnrs = []
        # ssims = []

        # for p, g in zip(preds, gt):
        #     p = torch.from_numpy(p).float().unsqueeze(0).permute(0, 3, 1,2)/255
        #     g = torch.from_numpy(g).float().unsqueeze(0).permute(0, 3, 1,2)/255
        #     psnrs.append(psnr(p, g))
        #     ssims.append(ssim(p, g))
            
        # overall_psnr.append(np.average(psnrs))
        # overall_ssim.append(np.average(ssims))
        # print(overall_psnr, overall_ssim)

        create_video(gt, preds, duration=10)
        sys.exit()

        
plt.figure(figsize=(6,6), dpi=300)
plt.plot(overall_psnr)
plt.xlabel(f'Epochs {range_tuple}'); plt.ylabel('PSNR')
plt.savefig('psnr_shell_drop_ADAMW_alpha_test_20.jpg')
plt.close()

plt.figure(figsize=(6,6), dpi=300)
plt.plot(overall_ssim)
plt.xlabel(f'Epochs {range_tuple}'); plt.ylabel('SSIM')
plt.savefig('ssim_shell_drop_ADAMW_alpha_test_20.jpg')
plt.close() 