"""
Script to learn MDP model from data for offline policy optimization
"""

import argparse
from collections import defaultdict
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from torch import nn
from torch.nn import functional as F
import torch

from models.nn_dynamics import WorldModel
from encoder import StateEncoder
from utils import seed, parse_config, load_data, save_config_yaml

import math

def construct_parser():
    parser = argparse.ArgumentParser(description='Training Dynamic Functions.')
    parser.add_argument("config_path", help="Path to config file")
    return parser


def plot_loss(train_loss, fn, xbar=None, title='Train Loss'):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    if xbar:
        ax.axhline(xbar, linestyle="--", color="black")
    ax.plot(train_loss)
    fig.savefig(fn)


def save_loss(train_loss, folder_name, prediction_error, model_name=None, eval_loss=None):
    model_name = f"{model_name}_" if model_name else ""
    for loss_name, losses in train_loss.items():
      fn_prefix = os.path.join(folder_name, f'{model_name}train_{loss_name}')
      plot_loss(losses, fn_prefix + '.png', title=loss_name)
      with open(fn_prefix+'.txt', 'w') as f:
        _l = np.array2string(np.array(losses), formatter={'float_kind':lambda x: "%.6f\n" % x})
        f.write(_l)
    with open(os.path.join(folder_name, f'{model_name}statistics.txt'), 'w') as f:
      f.write(f'Avg Prediction Error (unnormalized) {prediction_error:.16f}')

def plot_lipschitz_dist(lipschitz_coeff, folder_name, model_name=None, lipschitz_constraint=None):
    model_name = f"{model_name}_" if model_name else ""
    fig = plt.figure()
    ax = fig.add_subplot()
    max_local_L = np.max(lipschitz_coeff)
    fig.suptitle("Local Lipschitz Coefficient Distribution", x=0.62)
    ax.set_xlim(1, math.ceil(max_local_L * 2) / 2)
    ax.set_xticks([1, math.ceil(max_local_L * 2) / 2])
    ax.set_xlabel("Local Lipschitz Coefficient")
    ax.set_ylabel("Proportion of Labels")
    ax.spines[['right', 'top']].set_visible(False)
    ax.hist(lipschitz_coeff, bins=50, weights=np.ones(len(lipschitz_coeff)) / len(lipschitz_coeff))
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(folder_name, f"{model_name}train_local_lipschitz.png"), dpi=300)
    fig.savefig(os.path.join(folder_name, f"{model_name}train_local_lipschitz.pdf"), format="pdf", transparent=True)

def plot_err_dist(err_norms, folder_name, model_name=None):
    model_name = f"{model_name}_" if model_name else ""
    fig = plt.figure()
    ax = fig.add_subplot()
    err_min, err_max = np.min(err_norms), np.max(err_norms)
    fig.suptitle(f"Model Error Distribution (min={err_min:.3f}, max={err_max:.3f})")
    ax.hist(err_norms, density=False, bins=50)
    path = os.path.join(folder_name, f"{model_name}train_err_dist.png")
    fig.savefig(path)

def exists_prev_output(output_folder, config):
    f1 = os.path.join(output_folder, "dynamics.pkl")
    f2 = os.path.join(output_folder, "statistics.txt")
    return os.path.exists(f1) and os.path.exists(f2)

def list_of_dict_to_dict_of_list(a_list, divisor=1.0):
    if len(a_list) == 0:
        return []
    keys = a_list[0].keys()
    new_list = {k:[] for k in keys}
    for a_dict in a_list:
        for k in keys:
            new_list[k].append(a_dict[k] / divisor)
    return new_list

def encode_dataset(config, state_enc: StateEncoder, output_folder: str):
    with open(config.data.pkl, "rb") as f:
        data = pickle.load(f)
    enc_data = []
    for traj in data:
        s_vec = torch.from_numpy(np.array(traj["observations"])).float().cuda()
        s_img = torch.from_numpy(np.array(traj["img_observations"])).float().cuda()
        a = traj["actions"]
        s_enc = state_enc(s_vec, s_img)
        enc_data.append({
            "observations": s_enc.detach().cpu().numpy(),
            "actions": a
        })
    with open(os.path.join(output_folder, "encoded_dataset.pkl"), "wb") as f:
        pickle.dump(enc_data, f)

class MLPPolicy(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, layers: list):
        super().__init__()
        policy_layers = layers + [action_dim]
        self.layers = nn.Sequential(nn.Linear(latent_dim, policy_layers[0]))
        for i in range(len(policy_layers)-1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(policy_layers[i], policy_layers[i+1]))

        self.register_buffer("a_shift", torch.zeros(action_dim))
        self.register_buffer("a_scale", torch.ones(action_dim))

    def forward(self, x):
        a = self.layers(x)
        a = a * (self.a_scale + 1e-6) + self.a_shift
        return a

    def set_transformations(self, a_shift, a_scale):
        self.a_shift.copy_(a_shift)
        self.a_scale.copy_(a_scale)


def main():
    arg_parser = construct_parser()
    config = parse_config(arg_parser)
    output_folder = config.output.dynamics
    os.makedirs(output_folder, exist_ok=True)
    print(config)

    if exists_prev_output(output_folder, config) and not config.overwrite:
        print(f"Found existing results in {output_folder}, quit")
        exit(0)

    seed(config.seed)

    # Load Data
    s, a, sp = load_data(config, encoded=False)

    s_vec, s_img = [torch.from_numpy(x).float().cuda() for x in s]
    sp_vec, sp_img = [torch.from_numpy(x).float().cuda() for x in sp]
    a = torch.from_numpy(a).float().cuda()

    latent_dim = config.state_encoder.latent_dim

    # Construct Dynamics Model
    d_config = config.dynamics
    dynamics = WorldModel(latent_dim, a.shape[1], d_config=d_config,
                          hidden_size=d_config.layers,
                          fit_lr=d_config.lr,
                          fit_wd=d_config.weight_decay,
                          device="cpu" if config.no_gpu else "cuda",
                          activation=d_config.activation)
    
    state_enc = StateEncoder(latent_dim, vec_dim=s[0].shape[-1], fc_layers=config.state_encoder.layers).cuda()
    state_enc.fit_scaler(s_vec, s_img)
    target_state_enc = StateEncoder(latent_dim, vec_dim=s[0].shape[-1], fc_layers=config.state_encoder.layers).cuda()
    target_state_enc.fit_scaler(s_vec, s_img)
    target_state_enc.load_state_dict(state_enc.state_dict())

    a_shift = torch.mean(a, dim=0)
    a_scale = torch.mean(torch.abs(a - a_shift), dim=0)
    dynamics.dynamics_net.set_transformations(a_shift=a_shift, a_scale=a_scale)

    policy = MLPPolicy(latent_dim, a.shape[-1], config.state_encoder.policy_layers).cuda()
    policy.set_transformations(a_shift, a_scale)

    optimizer = torch.optim.Adam(list(state_enc.parameters()) + list(policy.parameters()) + list(dynamics.parameters()), lr=d_config.lr, weight_decay=d_config.weight_decay)

    batch_size = d_config.batch_size
    n_samples = a.shape[0]
    n_steps = n_samples // batch_size
    policy_loss_weight: float = config.state_encoder.policy_loss_weight
    tau = config.state_encoder.target_network_tau
    epoch_losses = []
    for _ in tqdm(range(d_config.train_epochs)):
        rand_idx = torch.LongTensor(np.random.permutation(n_samples)).cuda()
        ep_loss = defaultdict(float)

        for mb in range(n_steps):
            data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
            batch_s_vec = s_vec[data_idx]
            batch_s_img = s_img[data_idx]
            batch_sp_vec = sp_vec[data_idx]
            batch_sp_img = sp_img[data_idx]
            batch_a = a[data_idx]

            optimizer.zero_grad()
            s_enc = state_enc(batch_s_vec, batch_s_img)
            with torch.no_grad():
                sp_enc = target_state_enc(batch_sp_vec, batch_sp_img)
            pred_sp_enc = dynamics.predict(s_enc, batch_a)

            batch_a_pred = policy(s_enc)
            policy_loss = F.mse_loss(batch_a_pred, batch_a)

            dynamics_loss = F.mse_loss(pred_sp_enc, sp_enc)
            loss = dynamics_loss + policy_loss_weight * policy_loss

            loss.backward()
            optimizer.step()

            ep_loss['loss'] += loss.item()
            ep_loss['dynamics_loss'] += dynamics_loss.item()
            ep_loss['policy_loss'] += policy_loss.item()
        epoch_losses.append(ep_loss)
        for target_param, param in zip(target_state_enc.parameters(), state_enc.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    train_loss = list_of_dict_to_dict_of_list(epoch_losses, n_steps)

    # Save Model and config
    save_config_yaml(config, os.path.join(output_folder, "config.yaml"))

    with open(os.path.join(output_folder, "dynamics.pkl"), "wb") as f:
        pickle.dump(dynamics, f)
    
    with open(os.path.join(output_folder, "state_enc.pkl"), "wb") as f:
        pickle.dump(state_enc, f)
    
    with open(os.path.join(output_folder, "policy.pkl"), "wb") as f:
        pickle.dump(policy, f)

    encode_dataset(config, state_enc, output_folder)

    # Report Validation Loss
    s_enc = state_enc(s_vec, s_img)
    pred_errs = dynamics.eval_prediction_error(s_enc, a, state_enc(sp_vec, sp_img), d_config.batch_size, reduce_err=False)

    # Save Training Loss
    save_loss(train_loss, output_folder, np.mean(pred_errs), eval_loss=None)
    plot_err_dist(pred_errs, output_folder)

    # Save distribution of local lipschitz coefficients over data
    local_L = dynamics.eval_lipschitz_coeff(s_enc, a, batch_size=1024)
    if d_config.lipschitz_type != "none":
        if "spectral_normalization" in d_config.lipschitz_type:
            lipschitz_constraint = d_config.lipschitz_constraint**(1 + len(d_config.layers))
        else:
            lipschitz_constraint = d_config.lipschitz_constraint
    else:
        lipschitz_constraint = None
    # with open(os.path.join(output_folder, f"local_L_{lipschitz_constraint}.pkl"), "wb") as f:
    #     pickle.dump(local_L, f)
    plot_lipschitz_dist(local_L, output_folder, lipschitz_constraint=lipschitz_constraint)

if __name__ == "__main__":
    main()
