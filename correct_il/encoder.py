import torch
from torch import nn
import torchvision
from typing import Callable, Optional, List

class StateEncoder(nn.Module):
    def __init__(self, latent_dim, vec_dim=0, fc_layers: Optional[List[int]]=None):
        super().__init__()
        resnet = torchvision.models.resnet18()
        resnet.fc = nn.Identity()
        self.img_enc = replace_bn_with_gn(resnet)
        self.vec_dim = vec_dim
        if fc_layers:
            self.fc = nn.Sequential(
                nn.Linear(512 + vec_dim, fc_layers[0]),
                *[nn.Linear(fc_layers[i], fc_layers[i+1]) for i in range(len(fc_layers)-1)]
            )
        else:
            self.fc = nn.Identity()
        self.head = nn.Linear(fc_layers[-1] if fc_layers else 512 + vec_dim, latent_dim)
        self.register_buffer('vec_shift', torch.zeros(vec_dim))
        self.register_buffer('vec_scale', torch.ones(vec_dim))
        self.register_buffer('img_shift', torch.tensor(0.))
        self.register_buffer('img_scale', torch.tensor(1.))

    def forward(self, s_vec, s_img):
        s_vec = (s_vec - self.vec_shift) / (self.vec_scale + 1e-6)
        s_img = (s_img - self.img_shift) / (self.img_scale + 1e-6)
        s_img = torch.movedim(s_img, -1, -3)
        if len(s_vec.shape) == 1:
            s_vec = s_vec.unsqueeze(0)
        if len(s_img.shape) == 3:
            s_img = s_img.unsqueeze(0)
        s_enc = torch.cat([s_vec, self.img_enc(s_img)], dim=1)
        s_enc = self.fc(s_enc)
        return self.head(s_enc)

    def fit_scaler(self, s_vec, s_img):
        self.vec_shift, self.img_shift = torch.mean(s_vec, dim=0), torch.mean(s_img, dim=0)
        self.vec_scale = torch.mean(torch.abs(s_vec - self.vec_shift), dim=0)
        self.img_scale = torch.mean(torch.abs(s_img - self.img_shift), dim=0)


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
        root_module: nn.Module,
        features_per_group: int = 16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features)
    )
    return root_module
