import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class SinCosEmbedding(nn.Module):
    def __init__(self, 
                 n_levels,
                 in_dim,
                 include_input=False,
                 ):
        super().__init__()
        self.n_levels = n_levels
        self.in_dim = in_dim
        self.include_input = include_input

        self.freqs = torch.arange(n_levels, dtype=torch.float32)
        self.freqs = 2 ** self.freqs

        self.out_dim = in_dim * n_levels * 2
        if self.include_input:
            self.out_dim += in_dim

    def forward(self, x):
        """
        x: (*, in_dim)
        return: (*, out_dim)
        """
        out = []

        if self.include_input:
            out.append(x)

        for freq in self.freqs:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))

        out = torch.cat(out, dim=-1)
        
        assert out.shape[-1] == self.out_dim

        return out

ACT_NAME_TO_CLASS = {
    "relu": partial(nn.ReLU, inplace=True),
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "none": nn.Identity,
    'None': nn.Identity,
    None: nn.Identity,
}


def create_hiddens(n_hiddens, n_layers, activation='relu'):
    try:
        activation = ACT_NAME_TO_CLASS[activation]
    except KeyError:
        raise ValueError(f"Unknown activation: {activation}")

    hiddens = []
    for _ in range(n_layers):
        hiddens.append(nn.Linear(n_hiddens, n_hiddens))
        hiddens.append(activation())

    return hiddens

def create_head(n_hiddens, n_out, n_layers, activation='relu', last_activation=None):
    try:
        activation = ACT_NAME_TO_CLASS[activation]
    except KeyError:
        raise ValueError(f"Unknown activation: {activation}")

    hiddens = []
    for _ in range(n_layers-1):
        hiddens.append(nn.Linear(n_hiddens, n_hiddens))
        hiddens.append(activation())

    hiddens.append(nn.Linear(n_hiddens, n_out))
    hiddens.append(activation())

    return hiddens

class MeshCaster(nn.Module):
    def __init__(self, 
                n_verts,
                n_chan,
                n_mesh,
                n_levels_view,
                depth_view,
                depth_alpha,
                depth_attach,
                depth_head_alpha,
                depth_head_color,
                view_embedding='sincos',
                hidden_activation='relu',
                alpha_activation='none',
                color_activation='none',
                init_embeddings=False,
                init_hiddens=False,

                ):

        super().__init__()
        self.n_verts = n_verts
        self.n_chan = n_chan
        self.n_mesh = n_mesh
        self.n_levels_view = n_levels_view
        self.depth_view = depth_view
        self.depth_alpha = depth_alpha
        self.depth_attach = depth_attach
        self.depth_head_alpha = depth_head_alpha
        self.depth_head_color = depth_head_color
        self.hidden_activation = hidden_activation
        self.alpha_activation = alpha_activation
        self.color_activation = color_activation
        

        # View embedding
        if view_embedding == 'sincos':
            self.view_embed = SinCosEmbedding(n_levels=n_levels_view, in_dim=3)
        elif view_embedding == 'mlp':
            self.view_embed = nn.Sequential(
                nn.Linear(3, n_levels_view * 3 * 2),
                nn.ReLU(inplace=True),
            )

        # View embedding projection
        self.view_embed_dim = n_levels_view * 3 * 2
        self.view_embed_projector = nn.Linear(self.view_embed_dim, self.n_chan)

        # Vertex Embedding
        self.vertex_embeddings = nn.ModuleList(
            [nn.Embedding(self.n_verts+1, self.n_chan, max_norm=1.0) for _ in range(self.n_mesh)]
        )

        # hiddens
        self.view_hiddens = nn.Sequential(*create_hiddens(
            n_hiddens=self.n_chan, n_layers=self.depth_view, activation=self.hidden_activation))
        self.vertex_hiddens = nn.Sequential(*create_hiddens(
            n_hiddens=self.n_chan, n_layers=self.depth_alpha, activation=self.hidden_activation))

        # heads
        self.alpha_head = nn.Sequential(*create_head(
            n_hiddens=self.n_chan, n_out=1, n_layers=self.depth_head_alpha,
            activation=self.alpha_activation, last_activation=True))
        self.color_head = nn.Sequential(*create_head(
            n_hiddens=2*self.n_chan, n_out=3, n_layers=self.depth_head_color,
            activation=self.color_activation, last_activation=True))

        if init_embeddings:
            for m in self.vertex_embeddings:
                nn.init.normal_(m.weight, mean=0, std=self.n_chan ** -0.5)

        if init_hiddens:
            self.apply(self.init_hiddens)
    
    def init_hiddens(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self,
                verts,
                barys,
                views,
                ):
        """
        verts: tensors of shape (n_samples, n_mesh, 3)
        barys: tensors of shape (n_samples, n_mesh, 3) 
        views: tensors of shape (n_samples, n_mesh, 3) # TODO: change to (n_samples, 3) and use broadcast
        return: RGBA (n_samples, n_mesh, 4)
        """
        
        verts = verts + 1 # offset as zero is background
        n_mesh = verts.shape[1]

        if n_mesh != self.n_mesh:
            raise ValueError(f"n_mesh {n_mesh} does not match {self.n_mesh}")

        # view embedding
        # views (n_samples, n_mesh, 3) 
        view_embed = self.view_embed(views) # (n_samples, n_mesh, view_embed_dim)
        view_embed = self.view_embed_projector(view_embed) # (n_samples, n_mesh, n_chan)
        view_embed = self.view_hiddens(view_embed) # (n_samples, n_mesh, n_chan)

        # vertex embedding
        barys = barys.unsqueeze(-1) # (n_samples, n_mesh, 3, 1)
        vertex_embeds = []
        for i in range(n_mesh):
            vert_embeds = self.vertex_embeddings[i](verts[:, i, :].long()) # (n_samples, 3, n_chan)
            interp_embeds = torch.sum(barys[:, i, :, :] * vert_embeds, dim=1) # (n_samples, n_chan)
            vertex_embeds.append(interp_embeds.unsqueeze(1)) # (n_samples, 1, n_chan)

        vertex_embeds = torch.cat(vertex_embeds, dim=1) # (n_samples, n_mesh, n_chan)

        # get the outputs
        alphas = self.alpha_head(self.vertex_hiddens(vertex_embeds.clone())) # (n_samples, n_mesh, 1)

        # view_embed = view_embed.repeat(1, n_mesh, 1) # (n_samples, n_mesh, n_chan) # TODO: broadcast
        combined_embed = torch.cat([view_embed, vertex_embeds], dim=-1) # (n_samples, n_mesh, 2*n_chan)
        colors = self.color_head(combined_embed) # (n_samples, n_mesh, 3)

        # combine and return
        rgba = torch.cat([colors, alphas], dim=-1) # (n_samples, n_mesh, 4)
        return rgba

def front_to_back_compositing(rgba, masks=None, white_background=False, return_weights=False):
    """
        Args
        rgba: The RGBA values from closest to furthest (n_samples, n_mesh, 4) 
        masks: The masks for each mesh (n_samples, n_mesh, 1)
        Returns:
        rgb: The composited RGB values (n_samples, 3)
        weights: The weights of each mesh (n_samples, n_mesh)
    
    """

    alphas = rgba[:, :, -1:] # (n_samples, n_mesh, 1)
    colors = rgba[:, :, :-1] # (n_samples, n_mesh, 3)
    
    # restrict range to 0,1
    alphas = torch.sigmoid(alphas)
    colors = torch.sigmoid(colors)

    # regions that are not hit should not contribute to the final image
    # but are instead composited with the background
    if masks is not None:
        alphas = alphas * masks

    opacities = 1-alphas
    opacities = torch.cumprod(opacities, dim=1) # (n_samples, n_mesh, 1)
    last_opacity = opacities[:, -1:, :] # (n_samples, 1, 1)
    opacities = torch.cat([torch.ones_like(opacities[:, :1, :]), opacities[:, :-1, :]], dim=1) # (n_samples, n_mesh, 1)

    weights = alphas * opacities # (n_samples, n_mesh, 1)

    # multiplication is (n_samples, n_mesh, 1) * (n_samples, n_mesh, 3) = (n_samples, n_mesh, 3)
    # Then sum over the mesh dimension
    rgb = torch.sum(weights * colors, dim=1) # (n_samples, 3)

    if white_background:
        rgb = rgb + (1-torch.sum(weights, dim=1))

        # rgb = rgb + (1-torch.sum(last_opacity, dim=-1))

    if return_weights:
        return rgb, weights
    return rgb


if __name__ == '__main__':
    # embedder = SinCosEmbedding(n_levels=3, in_dim=2, include_input=False)

    # x = torch.randn(56, 2)
    # y = embedder(x)
    # assert y.shape == (56, 12)

    # x = torch.randn(1, 56, 2)
    # y = embedder(x)
    # assert y.shape == (1, 56, 12)

    # embedder = SinCosEmbedding(n_levels=3, in_dim=3, include_input=True)

    # x = torch.randn(56, 3)
    # y = embedder(x)
    # assert y.shape == (56, 21)

    # x = torch.randn(1, 56, 3)
    # y = embedder(x)
    # assert y.shape == (1, 56, 21)
    
    # print('test passed')


    # embedder = SinCosEmbedding(n_levels=3, in_dim=1, include_input=False)

    # x = torch.linspace(0, 2*3.14, 300).reshape(300, 1)
    # y = embedder(x)

    # import matplotlib.pyplot as plt
    # plt.plot(x[:].squeeze().numpy(), y[:, 0].squeeze().numpy(), label='sin')
    # plt.plot(x[:].squeeze().numpy(), y[:, 1].squeeze().numpy(), label='cos')
    # plt.plot(x[:].squeeze().numpy(), y[:, 2].squeeze().numpy(), label='sin2')
    # plt.plot(x[:].squeeze().numpy(), y[:, 3].squeeze().numpy(), label='cos2')
    # plt.plot(x[:].squeeze().numpy(), y[:, 4].squeeze().numpy(), label='sin3')
    # plt.plot(x[:].squeeze().numpy(), y[:, 5].squeeze().numpy(), label='cos3')
    # plt.legend()
    # plt.savefig('sin_cos.png')


    model = MeshCaster(n_verts = 570000,
                n_chan=16,
                n_mesh=6,
                n_levels_view=6,
                depth_view=4,
                depth_alpha=4,
                depth_attach=3,
                depth_head_alpha=1,
                depth_head_color=1,
                view_embedding='sincos',
                hidden_activation='relu',
                alpha_activation='none',
                color_activation='none',
                init_embeddings=True,
                init_hiddens=True,
                ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()

    ckpt = torch.load('model_sheep_mask_large__offset200_6.pth')
    model.load_state_dict(ckpt)

    from PIL import Image

    # Usual one
    @torch.no_grad()
    def sample_to_dict(sample):
        is_hit_mask = sample['is_hit_mask']
        img = sample['img']    
        
        data_dict = {
        'vert_idx_img': torch.from_numpy(sample['vert_idx_img']).cuda().unsqueeze(0).long().detach(),
        'bary_img': torch.from_numpy(sample['bary_img']).cuda().unsqueeze(0).float().detach(),
        'is_hit_mask': torch.from_numpy(sample['is_hit_mask']).cuda().unsqueeze(0).unsqueeze(-1).float().detach(),
        'img':torch.from_numpy(img[:, :, :]).cuda().unsqueeze(0).float().detach(),
        'view_dir':  torch.from_numpy(sample['view_dir']).cuda().unsqueeze(0).float().detach(),
        'depth_img': torch.from_numpy(sample['depth_img']).cuda().unsqueeze(0).unsqueeze(-1).float().detach(),

        }

        return data_dict

    import torch
    import random
    from tqdm import trange
    import numpy as np



    # neus 500k shell
    base_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/data/editable_nerf/blender_complex_textured_sphere/raster_base/test/r_'
    base_test_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/data/editable_nerf/blender_complex_textured_sphere/raster_base/test/r_'
    suffix = '.pth'

    b_in = base_path.replace('raster_base', 'raster_in0.03')
    b_in_test = base_test_path.replace('raster_base', 'raster_in0.03')

    b_off = base_path.replace('raster_base', 'raster_out0.03')
    b_off_test = base_test_path.replace('raster_base', 'raster_out0.03')

    base_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/data/editable_nerf/neus_sheep/raster_512_depth_renew_base/test/r_'
    base_test_path = '/datawaha/cggroup/parawr/Projects/adobe/neumesh/data/editable_nerf/neus_sheep/raster_512_depth/test/r_'
    suffix = '.pth'

    b_in_01 = base_path.replace('raster_512_depth_renew_base', 'raster_512_depth_renew_base')
    b_in = base_path.replace('raster_512_depth_renew_base', 'raster_512_depth_renew_in0.001')
    # b_in_test = base_test_path.replace('raster_512_depth_renew', 'raster_512_depth_renew_in0.001')

    b_off_01 = base_path.replace('raster_512_depth_renew_base', 'raster_512_depth_renew_out0.030')
    b_off = base_path.replace('raster_512_depth_renew_base', 'raster_512_depth_renew_out0.032')
    b_off_tt = base_path.replace('raster_512_depth_renew_base', 'raster_512_depth_renew_out0.020')
    # b_off_test = base_test_path.replace('raster_512_depth_renew', 'raster_512_depth_renew_out0.001')

    base_path = base_path.replace('raster_512_depth_renew_base', 'raster_512_depth_renew_out0.028')


    print('loading data')
    train_dataset = [torch.load(base_path + str(i) + suffix) for i in range(100, 200)]
    # train_dataset = [torch.load(base_test_path + str(i) + suffix) for i in range(100,200)]
    print(1)
    bin_dataset = [torch.load(b_in + str(i) + suffix) for i in range(100, 200)]
    bin_dataset_01 = [torch.load(b_in_01 + str(i) + suffix) for i in range(100, 200)]
    # bin_dataset = [torch.load(b_in_test + str(i) + suffix) for i in range(100,200)]
    print(2)

    boff_dataset = [torch.load(b_off + str(i) + suffix) for i in range(100, 200)]
    boff_dataset_01 = [torch.load(b_off_01 + str(i) + suffix) for i in range(100, 200)]
    boff_dataset_tt = [torch.load(b_off_tt + str(i) + suffix) for i in range(100, 200)]
    # boff_dataset = [torch.load(b_off_test + str(i) + suffix) for i in range(100,200)]
    print(3)


    train_dict = [sample_to_dict(i) for i in train_dataset[:100]]
    # test_dict = [sample_to_dict(i) for i in test_dataset]

    bin_dict = [sample_to_dict(i) for i in bin_dataset[:100]]
    bin_dict_01 = [sample_to_dict(i) for i in bin_dataset_01[:100]]
    # bin_test_dict = [sample_to_dict(i) for i in bin_test_dataset]

    boff_dict = [sample_to_dict(i) for i in boff_dataset[:100]]
    boff_dict_01 = [sample_to_dict(i) for i in boff_dataset_01[:100]]
    boff_dict_tt = [sample_to_dict(i) for i in boff_dataset_tt[:100]]

    def collate_fn(in_list):
        keys = in_list[0].keys()
        ret_dict = dict()
        for k in keys:
            if k not in [ 'img', 'view_dir']:
                ret_dict[k] = torch.cat([ll[k] for ll in in_list], dim=-1)
            else:
                ret_dict[k] = in_list[0][k]
        return ret_dict

    def collate_fn(in_list):
        keys = in_list[0].keys()
        ret_dict = dict()
        for k in keys:
            if k in [ 'vert_idx_img', 'view_dir', 'bary_img']:
                
                ret_dict[k] = torch.stack([ll[k].view(-1,3) for ll in in_list], dim=1)
            elif k in ['is_hit_mask', 'depth_img']:
                ret_dict[k] = torch.stack([ll[k].view(-1,1).bool() for ll in in_list], dim=1)
            else:
                ret_dict[k] = in_list[0][k]
        return ret_dict

    print('=' *10)
    print(model)
    print('=' *10)

    import matplotlib.pyplot as plt

    all_losses = []
    for i in range(1):
    # with torch.no_grad():
        losses = []
        for idx in trange(len(train_dict)):
            data_dict = collate_fn([boff_dict[idx], boff_dict_01[idx], train_dict[idx], boff_dict_tt[idx], bin_dict_01[idx], bin_dict[idx]])
            # data_dict = collate_fn([boff_dict[idx], train_dict[idx], bin_dict[idx]])

            optimizer.zero_grad()
            out = model(verts=data_dict['vert_idx_img'],
                        barys=data_dict['bary_img'],
                        views=data_dict['view_dir'],)

            rgb, weights = front_to_back_compositing(out, masks=data_dict['is_hit_mask'], white_background=True, return_weights=True)
            rgb = rgb.reshape((800,800,3))
            # print(torch.unique(weights))

            loss = torch.mean((rgb - data_dict['img'].squeeze())**2)
            if i == 0:
                xx = weights[..., 0, :].reshape((800, 800))
                yy = weights[..., 1, :].reshape((800, 800))
                zz = weights[..., 2, :].reshape((800, 800))
                uu = torch.sum(weights, dim=-2).reshape((800, 800))

                plt.imshow(xx.clone().detach().cpu().numpy(), vmin=0, vmax=1); plt.colorbar()
                plt.savefig(f'test1/weights{idx}.png')
                plt.close()

                plt.imshow(yy.clone().detach().cpu().numpy(), vmin=0, vmax=1); plt.colorbar()
                plt.savefig(f'test2/weights{idx}.png')
                plt.close()

                plt.imshow(zz.clone().detach().cpu().numpy(), vmin=0, vmax=1); plt.colorbar()
                plt.savefig(f'test3/weights{idx}.png')
                plt.close()

                plt.imshow(uu.clone().detach().cpu().numpy(), vmin=0, vmax=1); plt.colorbar()
                plt.savefig(f'test4/weights{idx}.png')
                plt.close()

                plt.imshow(rgb.clone().detach().cpu().numpy())
                plt.savefig(f'test5/rgb{idx}.png')
                plt.close()
    
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            losses.append(loss.item())
            all_losses.append(loss.item())

        print(f'epoch {i} loss {np.mean(losses)}')
        print(f'epoch {i} psnr {-10*np.log10(np.mean(losses))}')
        

    # # # # print(np.mean(losses))
    # plt.plot(all_losses)
    # plt.savefig('loss_sheep_mask_large_offset200_6.png')
    # plt.close()

    # plt.imshow(rgb.clone().detach().cpu().numpy())
    # plt.savefig('rgb_sheep_mask_large_offset200_6.png')

    # torch.save(model.state_dict(), 'model_sheep_mask_large__offset200_6.pth')
    