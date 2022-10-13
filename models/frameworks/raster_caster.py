import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from functools import partial


class MLP(nn.Module):
    def __init__(self,
                in_chan,
                out_chan,
                hidden_chan,
                activation='relu'):
        super(MLP, self).__init__()

        if activation == 'relu':
            act = nn.ReLU()
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
                viewdirs_attach):
        # TODO:
        # can we embed the barycentric coordinates and modulate
        # instead of adding
        
        super().__init__()
        
        self.n_verts = n_verts
        self.n_chan = n_chan
        self.n_hidden = n_hidden
        self.depth = depth
        self.use_viewdirs = use_viewdirs
        self.viewdirs_depth = viewdirs_depth
        self.viewdirs_attach = viewdirs_attach


        # setup the embeddings 
        self.vertex_embeddings = nn.Embedding(self.n_verts+1, self.n_hidden)

        # set up the network
        self.hiddens = nn.ModuleList(
            [MLP(self.n_hidden, self.n_hidden, self.n_hidden) for _ in range(depth)]
        )
        
        if self.use_viewdirs:
            self.viewdirs_embeddings = nn.Linear(3, self.n_hidden)
            self.viewdirs_hidden = nn.ModuleList(
                [MLP(self.n_hidden, self.n_hidden, self.n_hidden) for _ in range(viewdirs_depth)]
        )
                

        self.head = MLP(self.n_hidden, self.n_chan, self.n_hidden)

    def forward(self, x):
        vert_idx = x['vert_idx_img'] # N x H x W x 3
        vert_idx = vert_idx + 1 # because the pad vertex is -1
        barys = x['bary_img'] # N x H x W x 3
        viewdirs = x['view_dir']
        # is_hit_mask = x['is_hit_mask'] # N x H x W

        vert_embeds = self.vertex_embeddings(vert_idx) # N x H x W x 3 x n_hidden
        vert_embeds = vert_embeds * barys.unsqueeze(-1) # N x H x W x 3 x n_hidden
        vert_embeds = vert_embeds.sum(dim=3) # N x H x W x n_hidden

        # print(vert_embeds.shape)

        vert_hiddens = vert_embeds
        if self.use_viewdirs:
            viewdirs_embeddings = self.viewdirs_embeddings(viewdirs)
            viewdirs_hiddens = viewdirs_embeddings
            for layer in self.viewdirs_hidden:
                viewdirs_hiddens = F.relu(layer(viewdirs_hiddens))
                
        for ii, layer in enumerate(self.hiddens):
            vert_hiddens = F.relu(layer(vert_hiddens))
            if (ii == self.viewdirs_attach) and self.use_viewdirs:
                vert_hiddens = vert_hiddens + viewdirs_hiddens
        
        rgb = self.head(vert_hiddens) # N x H x W x n_chan

        return rgb


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

