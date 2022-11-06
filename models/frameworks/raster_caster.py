import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from functools import partial


class Embedder(nn.Module):
    def __init__(self,
                 n_hidden):
        super(Embedder, self).__init__()

        self.n_hidden = n_hidden

    def forward(self, x):
        return x

# from https://github.com/nv-tlabs/ATISS/blob/6b46c1133e8f67927330cbc662898a79fb2b86cb/scene_synthesis/networks/base.py#L13
class FixedPositionalEncoding(nn.Module):
    def __init__(self, proj_dims, val=0.9):
        super().__init__()
        ll = proj_dims // 2
        exb = 2 * torch.linspace(0, ll-1, ll) / proj_dims
        self.sigma = 1.0 / torch.pow(val, exb)
        self.sigma = 2 * np.pi * self.sigma
        # self.sigma = self.sigma.view(

    def forward(self, x):
        ndim = x.ndim # BxHxWxC
        shape = (1,) * ndim
        sigma = self.sigma.view((shape + (-1,)))  # 1x1x1x1xproj_dims
        x = x.unsqueeze(-1)
        return torch.cat([
            torch.sin(x * self.sigma.to(x.device)),
            torch.cos(x * self.sigma.to(x.device))
        ], dim=-1)


class SinCosEmbedder(Embedder):
    def __init__(self, 
                 n_in,
                 n_levels,
                 n_hidden):
        super().__init__(n_hidden=n_hidden)
        # I am assuming a B x chan tensor which will be expanded to B x chan x n_hidden
        # we flatten to B x (chan x n_hidden) an MLP then changes it B x n_hidden
        self.proj = FixedPositionalEncoding(n_levels)
        self.mlp = nn.Linear(n_in*n_levels, n_hidden)

    def forward(self, x):
        x = self.proj(x) # B x C x n_hidden
        shape = x.shape
        x = x.view(shape[:-2] + (-1,)) # B x (C x n_hidden)
        x = self.mlp(x)
        return x

class SimpleEmbedder(Embedder):
    def __init__(self, 
                 n_vert,
                 n_hidden):
        super().__init__(n_hidden=n_hidden)
        self.n_vert = n_vert
        self.embedding = nn.Embedding(n_vert, n_hidden)

    def forward(self, x):
        return self.embedding(x)


class TransformerEmbedder(Embedder):
    def __init__(self,
                n_layer,
                n_head,
                n_hidden,
                activation='relu'
                ):
        super().__init__(n_hidden=n_hidden)
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_hidden = n_hidden

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_hidden,
            nhead=n_head,
            dim_feedforward=4*n_hidden,
            batch_first=True,
            activation=activation
            )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer
            )
    
    def forward(self, x):
        return self.transformer_encoder(x)


class MLP(nn.Module):
    def __init__(self,
                in_chan,
                out_chan,
                hidden_chan,
                activation='relu'):
        super(MLP, self).__init__()

        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            act = nn.LeakyReLU(negative_slope=1e-1)

        self.mlp = nn.Sequential(
            nn.Linear(in_chan, hidden_chan),
            act,
            nn.Linear(hidden_chan, hidden_chan),
            act,
            nn.Linear(hidden_chan, out_chan),
        )

    def forward(self, x):
        return self.mlp(x)

class RasterCaster(nn.Module):
    def __init__(self,
                n_verts,
                n_chan,
                n_hidden,
                depth,
                use_viewdirs,
                viewdirs_depth,
                viewdirs_attach,
                n_mesh=1,
                dropout_inners=False,
                dropout_type='partial',
                alpha_composite=False):
        # TODO:
        # can we embed the barycentric coordinates and modulate
        # instead of adding
        # dropout_type can be partial or mesh, in mesh we drop a whole mesh
        
        super().__init__()
        
        self.n_verts = n_verts
        self.n_chan = n_chan
        self.n_hidden = n_hidden
        self.depth = depth
        self.use_viewdirs = use_viewdirs
        self.viewdirs_depth = viewdirs_depth
        self.viewdirs_attach = viewdirs_attach
        self.n_mesh = n_mesh
        self.dropout_inners = dropout_inners
        self.dropout_type = dropout_type


        # setup the embeddings 
        self.vertex_embeddings = nn.ModuleList(
                    [nn.Embedding(self.n_verts+1, self.n_hidden, max_norm=1.0) for _ in range(self.n_mesh)]
                    )
        self.vertex_sin_embedder = nn.Identity() #SinCosEmbedder(self.n_hidden, 4, self.n_hidden)
        
        if self.n_mesh > 1:
            self.vertex_embeddings_projector = nn.Sequential(
                # WARNING
                MLP(1*self.n_hidden, self.n_hidden, self.n_hidden),
                nn.ReLU(inplace=True))
        if self.dropout_inners:
            self.dropout_layer = nn.Dropout(0.1)
            
        # set up the network
        self.hiddens = nn.ModuleList(
            [MLP(self.n_hidden, self.n_hidden, self.n_hidden) for _ in range(depth)]
        )
        
        if self.use_viewdirs:
            self.viewdirs_embeddings = SinCosEmbedder(3, 6, self.n_hidden)
            self.viewdirs_hidden = nn.ModuleList(
                [MLP(self.n_hidden, self.n_hidden, self.n_hidden) for _ in range(viewdirs_depth)]
        )
                

        self.head = MLP(self.n_hidden, self.n_chan, self.n_hidden)
        
        for m in self.vertex_embeddings:
            nn.init.normal_(m.weight, mean=0, std=self.n_hidden ** -0.5)
            
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, x):
        vert_idx = x['vert_idx_img'] # N x H x W x (3xnm) 
        vert_idx = vert_idx + 1 # because the pad vertex is -1
        barys = x['bary_img'] # N x H x W x (3xnm) 
        viewdirs = x['view_dir']
        # is_hit_mask = x['is_hit_mask'] # N x H x W
        
        N, H, W, _ = vert_idx.shape
        
        vert_embeds_list = []
        barys_list = []
        for ii, embedder in enumerate(self.vertex_embeddings):
            pre_sin_embed = embedder(vert_idx[..., ii*3:(ii+1)*3])
            barys_list.append(barys[..., ii*3:(ii+1)*3])
            if self.dropout_inners and (ii < 2):
                if self.dropout_type == 'mesh':
                    pre_sin_embed = torch.rand_like(pre_sin_embed, device=pre_sin_embed.device)
                elif self.dropout_type == 'partial':
                    pre_sin_embed = self.dropout_layer(pre_sin_embed)
                        
            sin_embed = self.vertex_sin_embedder(pre_sin_embed)
            vert_embeds_list.append(sin_embed)  # N x H x W x (3) x n_hidden

        # vert_embeds = torch.cat(vert_embeds_list, dim=-2) # N x H x W x (3xnm) x n_hidden
        # print(vert_embeds.shape)
            
        # vert_embeds = self.vertex_embeddings(vert_idx) # N x H x W x (3xnm) x n_hidden
        
        
        # vert_embeds = vert_embeds * barys.unsqueeze(-1) # N x H x W x (3xnm) x n_hidden
        n_mesh = vert_idx.shape[-1] // 3
        if self.n_mesh != n_mesh:
            raise ValueError('Supplied data does not match n_mesh')
            
        ####### WARNING, PLEASE CHANGE

        # vert_embeds = vert_embeds.view(N, H, W, n_mesh, 3, -1).contiguous()
        # vert_embeds = vert_embeds.sum(dim=4) # N x H x W x nm x n_hidden
        
        # vert_embeds = vert_embeds.view(N, H, W, -1).contiguous()

        if self.use_viewdirs:
            viewdirs_embeddings = self.viewdirs_embeddings(viewdirs)
            viewdirs_hiddens = viewdirs_embeddings
            for layer in self.viewdirs_hidden:
                viewdirs_hiddens = F.relu(layer(viewdirs_hiddens), inplace=True)
        
        # hiddens = []
        rgbas_list = []
        for barys, curr_vembed in zip(barys_list, vert_embeds_list):
            # print(curr_vembed.shape, barys.shape)
            vert_embeds = curr_vembed * barys.unsqueeze(-1)
            vert_embeds = vert_embeds.sum(3)
            if n_mesh > 1:
                vert_hiddens = self.vertex_embeddings_projector(vert_embeds)
            else:
                vert_hiddens = vert_embeds
            
                    
            for ii, layer in enumerate(self.hiddens):
                vert_hiddens = F.relu(layer(vert_hiddens), inplace=True)
                if (ii == self.viewdirs_attach) and self.use_viewdirs:
                    # print(vert_hiddens.shape, viewdirs_hiddens.shape)

                    vert_hiddens = vert_hiddens * viewdirs_hiddens


            # hiddens.append(vert_hiddens)
        
            rgb = self.head(vert_hiddens) # N x H x W x 4/3

            rgbas_list.append(rgb) 

        return rgbas_list

def prep_lists_for_compositing(in_list):
    """ given a list of outputs from shells, flatten them for compositing
    Args:
        in_list: List of len nm of tensors of shape NxHxWxC
    Returns:
        out_list: a single tensor of shape NHWx(nm)xC
    """
    # Assuming that the order is base, inside, out
    bb, ii, oo = in_list # each is N x H x W x C
    n_chan = bb.shape[-1]
    bb = bb.view(-1, n_chan).unsqueeze(-2) # NHW x 1 x C
    ii = ii.view(-1, n_chan).unsqueeze(-2) # NHW x 1 x C
    oo = oo.view(-1, n_chan).unsqueeze(-2) # NHW x 1 x C

    output_tensor = torch.cat([oo, bb, ii], dim=-2) # maintain channels NxHxWx(nm)xC
    output_tensor = output_tensor.squeeze(-1) # NHW x (nm)xC squeeze last channel
    return output_tensor

@torch.no_grad()
def compute_masks(zero, inside, outside):
    with torch.no_grad():
        no_hit = ~(zero.bool() | inside.bool() | outside.bool())
        one_hit = ~(zero.bool() | inside.bool() | ~outside.bool())
        two_hit = ~(~zero.bool() | inside.bool() | ~outside.bool())
        three_hit = ~(~zero.bool() | ~inside.bool() | ~outside.bool())

    return no_hit.detach(), one_hit.detach(), two_hit.detach(), three_hit.detach()

@torch.no_grad()
def correct_depths(is_hit_masks_lists, depths_list, bkgd_depths=(7, 7.25, 7.5), virtual_offset=0.03):
    """ Rays do not hit all meshes. We want to make a virtual intersection point just
        after the hit point with outside meshes
    Args:
        is_hit_masks_lists: list of nm masks of shape N x H x W
        depths_list: list of nm depths
    Returns:
        corrected_depths_list: list of nm depths, where the background is given a virtual intersection point
    """
    # the current order is base, in, out
    hit_masks = compute_masks(*is_hit_masks_lists)
    depths_list = [d.clone().detach() for d in depths_list]
    
    BASE_IDX = 0
    OUTER_IDX = 2
    INSIDE_IDX = 1

    NO_HIT_IDX = 0
    ONE_HIT_IDX = 1
    TWO_HIT_IDX = 2
    THREE_HIT_IDX = 3

    # No hit adjustments
    depths_list[OUTER_IDX][hit_masks[NO_HIT_IDX]] = bkgd_depths[0] # out -> closest
    depths_list[BASE_IDX][hit_masks[NO_HIT_IDX]] = bkgd_depths[1] # base -> middle
    depths_list[INSIDE_IDX][hit_masks[NO_HIT_IDX]] = bkgd_depths[2] # in -> farthest
    
    # one hit adjustments
    # outer has been hit, move others to virtual offset
    depths_list[BASE_IDX][hit_masks[ONE_HIT_IDX]] = depths_list[OUTER_IDX][hit_masks[ONE_HIT_IDX]] + virtual_offset
    depths_list[INSIDE_IDX][hit_masks[ONE_HIT_IDX]] = depths_list[OUTER_IDX][hit_masks[ONE_HIT_IDX]] + 2 * virtual_offset
    
    # Two hit adjustments
    # outer and base hit, move inner to base + virtual_offset
    depths_list[INSIDE_IDX][hit_masks[TWO_HIT_IDX]] = depths_list[BASE_IDX][hit_masks[TWO_HIT_IDX]] + virtual_offset

    return depths_list, hit_masks


# from https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf.py#L262
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    neg = dists < 0
    dists[neg] = -dists[neg]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)  # [N_rays, N_samples]

    # dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    # if raw_noise_std > 0.:
    #     noise = torch.randn(raw[...,3].shape) * raw_noise_std

    #     # Overwrite randomly sampled data if pytest
    #     if pytest:
    #         np.random.seed(0)
    #         noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
    #         noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(dists.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def alpha_composite_rgbs(rgba_list):
    """ alpha composite a list of rgbs 
    """
    alphas = torch.stack([rgba[..., 3] for rgba in rgba_list], dim=-1)
    alphas = F.relu(alphas)
    rgbs = torch.stack([rgba[..., :3] for rgba in rgba_list], dim=-1)
    rgb = torch.sigmoid(rgbs)
    # alphas = alphas / alphas.sum(dim=-1, keepdim=True)
    rgb = (rgbs * alphas.unsqueeze(-2)).sum(dim=-1) # -2 as want alpha to be broadcast over rgb, NOT n_mesh
    alphas = alphas.sum(dim=-1, keepdim=True)
    return rgb, alphas


def get_raster_caster_model(args):
    pass


if __name__ == '__main__':
    model = RasterCaster(n_verts = 100,
                      n_chan=3,
                      n_hidden=128,
                      depth=4)

    print(model)

    # --------------------------------------------
    # generate input data
    # --------------------------------------------
    B, H, W = 4, 200, 200
    verts = torch.randint(0, 100, (B, H, W, 3)) # N x H x W x 3
    barys = torch.rand((B, H, W, 3)) # N x H x W x 3
    is_hit_mask = torch.ones((B, H, W)) # N x H x W x 3

    x = {
        'vert_idx_img': verts,
        'bary_img': barys,
        'is_hit_mask': is_hit_mask
    }

    out = model(x)

    print('out shape: {}'.format(out.shape))

