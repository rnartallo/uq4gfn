from matplotlib import pyplot as plt
from torch.distributions import Normal, Categorical, MultivariateNormal
import math
import numpy as np
from scipy.stats import norm
import torch
import random
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import colormaps as cm
import tqdm
import torch.optim as optim
from collections import Counter
import networkx as nx
from collections import deque
from tqdm import tqdm
import pandas as pd
import random
from matplotlib.colors import LogNorm
if not hasattr(np, 'bool'):
    np.bool = bool
import chaospy as chaos
from matplotlib.collections import LineCollection

trajectory_length = 5
min_policy_std = 0.1
max_policy_std = 1.0
batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PlaneEnvironment():
    def __init__(self, mus, variances, n_sd, init_value):
        self.mus = torch.tensor(mus)
        self.sigmas = torch.tensor([math.sqrt(v) for v in variances])
        self.variances = torch.tensor(variances)
        self.mixture = [
            MultivariateNormal(torch.tensor(m), s*torch.eye(2)) for m, s in zip(self.mus, self.sigmas)
        ]

        self.n_sd = n_sd
        self.lbx = min(self.mus[:,0]) - self.n_sd *max(self.sigmas)
        self.ubx = max(self.mus[:,0]) + self.n_sd *max(self.sigmas)
        self.lby = min(self.mus[:,1]) - self.n_sd *max(self.sigmas)
        self.uby = max(self.mus[:,1]) + self.n_sd *max(self.sigmas)

        self.init_value = init_value 

    def log_reward(self, x,y):
        xy = torch.stack([x, y])
        return torch.logsumexp(torch.stack([m.log_prob(xy.T) for m in self.mixture]), 0)

    @property
    def log_partition(self) -> float:
        return torch.tensor(len(self.mus)).log()

def get_policy_dist(model, pos):
    pf_params = model(pos)  
    policy_means = pf_params[:, :2]
    
    stds = torch.sigmoid(pf_params[:, 2:]) * (max_policy_std - min_policy_std) + min_policy_std
    covar = torch.stack([torch.diag_embed(std**2) for std in stds], dim=0)  # [B, 2, 2]

    return torch.distributions.MultivariateNormal(policy_means, covar)

def step(pos, action):
    new_pos = torch.zeros_like(pos)
    new_pos[:, :2] = pos[:, :2] + action  # move
    new_pos[:, 2] = pos[:, 2] + 1  # increment step counter

    return new_pos


def initalize_state(batch_size, device, env, randn=False):
    # Trajectory starts at state (X_0,Y_0, 0) 
    pos = torch.zeros((batch_size, 3), device=device)
    pos[:, :2] = env.init_value

    return pos

def inference(trajectory_length, forward_model, env, batch_size=20000):
    """Sample some trajectories."""

    with torch.no_grad():
        trajectory = torch.zeros((batch_size, trajectory_length + 1, 3), device=device)
        trajectory[:, 0, :2] = env.init_value  # Set x,y initial position
        trajectory[:, 0, 2] = 0 

        x = initalize_state(batch_size, device, env)

        for t in range(trajectory_length):
            policy_dist = get_policy_dist(forward_model, x)
            action = policy_dist.sample()

            new_x = step(x, action)
            trajectory[:, t + 1, :] = new_x
            x = new_x

    return trajectory

def setup_experiment(hid_dim=100, lr_model=1e-3, lr_logz=1e-1):
    
    # input : [x_position, y_position, n_steps], output = [mus, stds].
    forward_model = torch.nn.Sequential(torch.nn.Linear(3, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, hid_dim),
                                        torch.nn.ELU(),
                                        torch.nn.Linear(hid_dim, 4)).to(device)

    backward_model = torch.nn.Sequential(
        torch.nn.Linear(3, hid_dim),         # input: x,y,t
        torch.nn.ELU(),
        torch.nn.Linear(hid_dim, hid_dim),
        torch.nn.ELU(),
        torch.nn.Linear(hid_dim, 4)          # output: mean x, std x, mean y, std y
    ).to(device)

    logZ = torch.nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = torch.optim.Adam(
        [
            {'params': forward_model.parameters(), 'lr': lr_model},
            {'params': backward_model.parameters(), 'lr': lr_model},
            {'params': [logZ], 'lr': lr_logz},
        ]
    )

    return (forward_model, backward_model, logZ, optimizer)

def render(env,ax):
    x = torch.linspace(env.lbx, env.ubx, 100)
    y = torch.linspace(env.lby, env.uby, 100)
    xx, yy = torch.meshgrid(x, y, indexing='ij')  

    xy = torch.stack([xx.flatten(), yy.flatten()], dim=1) 
    log_probs = env.log_reward(xy[:, 0], xy[:, 1])  
    d = torch.exp(log_probs).reshape(100, 100)  

    ax.imshow(d.numpy(), origin='lower', extent=(env.lbx, env.ubx, env.lby, env.uby), aspect='auto', cmap = 'Reds',alpha = 1)
    ax.set_aspect('equal')
    ax.set_xticks([-2,0,2])
    ax.set_yticks([-2,0,2])
    return

def plot_traj(trajectory,ax,trajectory_length):
    t = np.linspace(0, trajectory_length, trajectory_length + 1)
    traj = trajectory.reshape(trajectory_length + 1, 3)

    sc = ax.scatter(traj[:, 0], traj[:, 1], c=t, cmap='viridis',zorder = 2 )

    points = traj[:, :2]
    segments = np.stack([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(t.min(), t.max()),zorder = 1)
    lc.set_array(t[:-1]) 
    lc.set_linewidth(2)

    ax.add_collection(lc)

    return sc, lc

def extract_flow_at_point(forward_model,point):
    policy_dist = get_policy_dist(forward_model,torch.tensor(point).to(device))
    x_mean, y_mean = policy_dist.loc[0,:].detach().cpu().numpy()
    x_var = policy_dist.covariance_matrix[0,0,0].detach().cpu().numpy()
    y_var = policy_dist.covariance_matrix[0,1,1].detach().cpu().numpy()
    x_dist = norm(x_mean,np.sqrt(x_var))
    y_dist = norm(y_mean,np.sqrt(y_var))
    return x_dist, y_dist

def extract_dist_at_point(forward_model,point):
    policy_dist = get_policy_dist(forward_model,torch.tensor(point).to(device))
    x_mean, y_mean = policy_dist.loc[0,:].detach().cpu().numpy()
    x_var = policy_dist.covariance_matrix[0,0,0].detach().cpu().numpy()
    y_var = policy_dist.covariance_matrix[0,1,1].detach().cpu().numpy()
    return x_mean, x_var, y_mean, y_var

def meta_inference(trajectory_length, forward_model_ensemble, env, batch_size=20000):

    with torch.no_grad():
        trajectory = torch.zeros((batch_size, trajectory_length + 1, 3), device=device)
        trajectory[:, 0, :2] = env.init_value  # Set x,y initial position
        trajectory[:, 0, 2] = 0 

        x = initalize_state(batch_size, device, env)

        for t in range(trajectory_length):
            forward_model = random.choice(forward_model_ensemble)
            policy_dist = get_policy_dist(forward_model, x)
            action = policy_dist.sample()

            new_x = step(x, action)
            trajectory[:, t + 1, :] = new_x
            x = new_x

    return trajectory.cpu().numpy()

def plot_outputs(outputs,trajectory):

    N_samples = outputs.shape[0]

    fig, ax = plt.subplots(2,5)
    fig.set_size_inches(15,6)

    all_supp_coords_x = []
    all_dens_coords_x = []

    all_supp_coords_y = []
    all_dens_coords_y = []

    for i in range(0,5):

        all_supp_coords_ix = []
        all_dens_coords_ix = []

        all_supp_coords_iy = []
        all_dens_coords_iy = []

        x_dist_mean = np.zeros(100)
        y_dist_mean = np.zeros(100)
        
        for n in range(0,N_samples):
            
            x_sup = np.linspace(-3, 3, 100)
            y_sup = np.linspace(-3, 3, 100)

            x_mean, x_logvar, y_mean, y_logvar = outputs[n,i,:]
            
            x_dist = norm(x_mean,np.sqrt(np.exp(x_logvar)))
            all_supp_coords_ix.extend(x_sup)
            all_dens_coords_ix.extend(x_dist.pdf(x_sup))

            x_dist_mean += x_dist.pdf(x_sup)/N_samples
            point = trajectory[0, i, :]
            x, y, _ = point
            ax[0,i].set_title(r'$(x,y) = $' + f"({x:.4f}, {y:.4f})")
            #ax[0,i].legend()
            next_point = trajectory[0, i+1, :]
            
            y_dist = norm(y_mean,np.sqrt(np.exp(y_logvar)))
            
            all_supp_coords_iy.extend(y_sup)
            all_dens_coords_iy.extend(y_dist.pdf(y_sup))
            
            y_dist_mean += y_dist.pdf(y_sup)/N_samples
            point = trajectory[0, i, :]
            x, y, _ = point
            ax[1,i].set_title(r'$(x,y) = $' + f"({x:.4f}, {y:.4f})")
            #ax[1,i].legend()
            next_point = trajectory[0, i+1, :]
        
        all_supp_coords_x.append(all_supp_coords_ix)
        all_dens_coords_x.append(all_dens_coords_ix)
        all_supp_coords_y.append(all_supp_coords_iy)
        all_dens_coords_y.append(all_dens_coords_iy)
        
        #ax[0,i].plot(x_sup,x_dist_mean,color = 'black')
        #ax[1,i].plot(y_sup,y_dist_mean,color = 'black')
        
        ax[0,i].axvline(next_point[0]-x,linestyle = 'dotted',color = 'k')
        ax[1,i].axvline(next_point[1]-y,linestyle = 'dotted',color = 'k')
        #ax[0,i].set_ylim([0,1])
        #ax[1,i].set_ylim([0,1])

        h = ax[0,i].hist2d(
        all_supp_coords_x[i], 
        all_dens_coords_x[i], 
        bins=(100, 100), 
        cmap='magma_r',
        cmin=1)
        ax[0,i].set_facecolor('gainsboro')

        h = ax[1,i].hist2d(
        all_supp_coords_y[i], 
        all_dens_coords_y[i], 
        bins=(100, 100), 
        cmap='magma_r',
        cmin=1)
        ax[1,i].set_facecolor('gainsboro')

        point = trajectory[0, i, :]
        x, y, _ = point
        ax[0,i].set_title(r'$(x,y) = $' + f"({x:.4f}, {y:.4f})")
        next_point = trajectory[0, i+1, :]

        x, y, _ = point
        ax[1,i].set_title(r'$(x,y) = $' + f"({x:.4f}, {y:.4f})")
        next_point = trajectory[0, i+1, :]
        
        ax[0,i].axvline(next_point[0]-x,linestyle = 'dotted',color = 'k')
        ax[1,i].axvline(next_point[1]-y,linestyle = 'dotted',color = 'k')


    ax[0,0].set_ylabel(r'Flow density in $x$-direction')
    ax[1,0].set_ylabel(r'Flow density in $y$-direction')

    return fig, ax 