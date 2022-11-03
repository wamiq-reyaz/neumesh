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
                dropout_type='partial'):
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
        
            rgb = self.head(vert_hiddens) # N x H x W x 4
            rgbas_list.append(rgb) 

        # rgbas = torch.cat(rgbas_list, dim=-2) # N x H x W x (3xnm) x n_chan

        return rgbas_list

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

