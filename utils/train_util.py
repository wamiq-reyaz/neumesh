import torch
import numpy as np
from typing import Iterable


def calc_grad_norm(norm_type=2.0, **named_models):
    gradient_norms = {"total": 0.0}
    for name, model in named_models.items():
        gradient_norms[name] = 0.0
        for p in list(model.parameters()):
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                gradient_norms[name] += param_norm.item() ** norm_type
        gradient_norms["total"] += gradient_norms[name]
    for k, v in gradient_norms.items():
        gradient_norms[k] = v ** (1.0 / norm_type)
    return gradient_norms


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def batchify_query(query_fn, *args: Iterable[torch.Tensor], chunk, dim_batchify):
    # [(B), N_rays, N_pts, ...] -> [(B), N_rays*N_pts, ...]
    _N_rays = args[0].shape[dim_batchify]
    _N_pts = args[0].shape[dim_batchify + 1]
    args = [arg.flatten(dim_batchify, dim_batchify + 1) for arg in args]
    _N = args[0].shape[dim_batchify]
    raw_ret = []
    for i in range(0, _N, chunk):
        if dim_batchify == 0:
            args_i = [arg[i : i + chunk] for arg in args]
        elif dim_batchify == 1:
            args_i = [arg[:, i : i + chunk] for arg in args]
        elif dim_batchify == 2:
            args_i = [arg[:, :, i : i + chunk] for arg in args]
        else:
            raise NotImplementedError
        raw_ret_i = query_fn(*args_i)
        if not isinstance(raw_ret_i, tuple):
            raw_ret_i = [raw_ret_i]
        raw_ret.append(raw_ret_i)
    collate_raw_ret = []
    num_entry = 0
    for entry in zip(*raw_ret):
        if isinstance(entry[0], dict):
            tmp_dict = {}
            for list_item in entry:
                for k, v in list_item.items():
                    if k not in tmp_dict:
                        tmp_dict[k] = []
                    tmp_dict[k].append(v)
            for k in tmp_dict.keys():
                # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
                # tmp_dict[k] = torch.cat(tmp_dict[k], dim=dim_batchify).unflatten(dim_batchify, [_N_rays, _N_pts])
                # NOTE: compatible with torch 1.6
                v = torch.cat(tmp_dict[k], dim=dim_batchify)
                tmp_dict[k] = v.reshape(
                    [
                        *v.shape[:dim_batchify],
                        _N_rays,
                        _N_pts,
                        *v.shape[dim_batchify + 1 :],
                    ]
                )
            entry = tmp_dict
        else:
            # [(B), N_rays*N_pts, ...] -> [(B), N_rays, N_pts, ...]
            # entry = torch.cat(entry, dim=dim_batchify).unflatten(dim_batchify, [_N_rays, _N_pts])
            # NOTE: compatible with torch 1.6
            v = torch.cat(entry, dim=dim_batchify)
            entry = v.reshape(
                [*v.shape[:dim_batchify], _N_rays, _N_pts, *v.shape[dim_batchify + 1 :]]
            )
        collate_raw_ret.append(entry)
        num_entry += 1
    if num_entry == 1:
        return collate_raw_ret[0]
    else:
        return tuple(collate_raw_ret)

# from https://github.com/NVlabs/stylegan3/blob/b1a62b91b18824cf58b533f75f660b073799595d/torch_utils/misc.py#L110
class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1