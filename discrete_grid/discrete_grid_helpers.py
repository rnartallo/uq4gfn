from matplotlib import pyplot as plt
from torch.distributions import Normal, Categorical
import math
import numpy as np
import torch
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import time
from pyro import poutine

grid_size = 10
high_reward = 200
mid_reward = 40
low_reward = 0.1

high_reward_centers = [(2, 2), (2, 7), (7, 2), (7, 7)]

def create_reward_grid(grid_size,high_reward_centers,high_reward,mid_reward,low_reward):
    grid = np.full((grid_size, grid_size), low_reward)

    for center_y, center_x in high_reward_centers:
        if 0 <= center_y < grid_size and 0 <= center_x < grid_size:
            grid[center_y, center_x] = high_reward

        neighbors = [
            (center_y - 1, center_x),
            (center_y + 1, center_x),
            (center_y, center_x - 1),
            (center_y, center_x + 1),
        ]

        for ny, nx in neighbors:
            if 0 <= ny < grid_size and 0 <= nx < grid_size:
                if grid[ny, nx] != high_reward:
                    grid[ny, nx] = mid_reward
    return grid

def shift_reward_centers(p,high_reward_centers = [(2, 2), (2, 7), (7, 2), (7, 7)]):
    shifts = [(0,1), (0,-1), (1,0), (-1,0)]
    new_high_reward_centers = []
    for ix, center in enumerate(high_reward_centers):
        u = np.random.random()
        if u<p:
            shift = random.choice(shifts)
            new_high_reward_centers.append(tuple(np.array(center) + np.array(shift)))
        else:
            new_high_reward_centers.append(center)
    return new_high_reward_centers

def create_one_hot_training_data(numpy_grids,grid_size):
    unique_values = [0.1, 40, 200]
    
    value_to_index = {value: i for i, value in enumerate(unique_values)}
    num_categories = len(unique_values)
    num_grids = len(numpy_grids)

    one_hot_data = np.zeros((num_grids, 10, 10, num_categories), dtype=np.float32)

    for i, grid in enumerate(numpy_grids):
        for r in range(grid_size):
            for c in range(grid_size):
                value = grid[r, c]
                index = value_to_index[value]
                one_hot_data[i, r, c, index] = 1.0


    training_data = torch.tensor(one_hot_data).float().permute(0, 3, 1, 2)
    
    return training_data


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = h.view(-1, 32 * 3 * 3)
        h = self.relu(self.fc1(h))
        return self.fc_mean(h), self.fc_log_var(h)

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 32 * 3 * 3)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        h = h.view(-1, 32, 3, 3)
        h = self.relu(self.deconv1(h))       # shape: (B, 16, 6, 6)
        logits = self.deconv2(h)             # shape: (B, 3, 10, 10)
        logits = logits.permute(0, 2, 3, 1)   # (B, 10, 10, 3)
        return logits.view(-1, 100, 3)        # (B, 100, 3)


class VAE(nn.Module):
    def __init__(self, latent_dim=2,beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
        self.beta = beta

    def model(self, x, x_indices):
        pyro.module("decoder", self.decoder)
        batch_size = x.size(0)

        z_loc = x.new_zeros((batch_size, self.latent_dim))
        z_scale = x.new_ones((batch_size, self.latent_dim))

        with pyro.plate("data", batch_size):
            with pyro.poutine.scale(scale=self.beta):
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            logits = self.decoder(z)  

            pyro.sample("obs", dist.Categorical(logits=logits).to_event(1), obs=x_indices)

    def guide(self, x, x_indices=None):
        pyro.module("encoder", self.encoder)
        z_mean, z_log_var = self.encoder(x)
        z_scale = torch.exp(0.5 * z_log_var)

        with pyro.plate("data", x.size(0)):
            with pyro.poutine.scale(scale=self.beta):
                pyro.sample("latent", dist.Normal(z_mean, z_scale).to_event(1))

def get_latent_representation(vae, grid_data,grid_size):
    prepared_grid = create_one_hot_training_data([grid_data],grid_size).reshape(3,10,10)
    z_mean, _ = vae.encoder(prepared_grid)
    return z_mean.detach().numpy()

def get_latent_representation_dist(vae, grid_data,grid_size):
    prepared_grid = create_one_hot_training_data([grid_data],grid_size).reshape(3,10,10)
    z_mean, z_logvar = vae.encoder(prepared_grid)
    return z_mean.detach().numpy(), z_logvar.detach().numpy()

def visualize_grid(grid,ax):
    cax = ax.imshow(grid, cmap='Reds')
    ax.invert_yaxis()
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(np.arange(grid_size))
    ax.set_yticklabels(np.arange(grid_size))
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    return

def plot_trajectory(ax,sampled_path,reward_grid,actions):
    grid_size = 10

    grid = np.zeros((grid_size, grid_size))
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2,zorder=1)

    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.set_xticklabels(np.arange(0, grid_size, 1))
    ax.set_yticklabels(np.arange(0, grid_size, 1))

    for idx, (x, y) in enumerate(sampled_path):
        grid[y, x] = idx + 1

        if idx == 0:
            ax.text(x-0.075,y-0.05,'S', c='black',fontsize=10,backgroundcolor = 'white')
        if idx == len(sampled_path)-1:
            R = reward_grid.reward_grid[y,x].detach().numpy()
            ax.text(x-0.2,y,r'$R=$' + f"{R:.0f}", c='black',fontsize=6,backgroundcolor = 'white')
        if idx<len(sampled_path)-1:
            if actions[idx] == 0:
                ax.arrow(x-0.25,y,-0.5,0,head_width = 0.1,fc='white',ec='white',length_includes_head=True,zorder=2)
            elif actions[idx] == 1:
                ax.arrow(x+0.25,y,0.5,0,head_width = 0.1,fc='white',ec='white',length_includes_head=True,zorder=2)
            elif actions[idx] == 2:
                ax.arrow(x,y-0.25,0,-0.5,head_width = 0.1,fc='white',ec='white',length_includes_head=True,zorder=2)
            elif actions[idx] == 3:
                ax.arrow(x,y+0.25,0,0.5,head_width = 0.1,fc='white',ec='white',length_includes_head=True,zorder=2)

    masked_grid = np.ma.masked_where(grid == 0, grid)

    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(vmin=1, vmax=len(sampled_path))

    c = ax.imshow(masked_grid, cmap=cmap, norm=norm, origin='lower')
    return

class GridEnvironment:

    def __init__(self, reward_grid):
        self.grid_size = reward_grid.shape[0]
        self.reward_grid = torch.tensor(reward_grid, dtype=torch.float32)
        self.action_space_size = 5 # Up, Down, Left, Right, Terminate

        self.inverse_action = {
            0: 1,  # Up <-> Down
            1: 0,
            2: 3,  # Left <-> Right
            3: 2,
            4: 4   # Terminate stays the same
        }

    def get_initial_state(self):
        return (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))

    def get_reward(self, state):
        y, x = state
        return self.reward_grid[y, x]

    def step(self, state, action):
        y, x = state
        if action == 0: y -= 1
        elif action == 1: y += 1
        elif action == 2: x -= 1
        elif action == 3: x += 1
        return (y, x)

    def get_valid_actions(self, state):

        y, x = state
        valid_actions = [] 
        if y > 0: valid_actions.append(0) # Down
        if y < self.grid_size - 1: valid_actions.append(1) # Up
        if x > 0: valid_actions.append(2) # Left
        if x < self.grid_size - 1: valid_actions.append(3) # Right
        if self.reward_grid[y,x] > low_reward:
            valid_actions.append(4)
        return valid_actions
        
    
    def step_backward(self, state, action):
        y, x = state
        if action == 0: y += 1
        elif action == 1: y -= 1
        elif action == 2: x += 1
        elif action == 3: x -= 1
        return (y, x)

class GFlowNet(nn.Module):
    def __init__(self, state_dim, action_space_size):
        super(GFlowNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_size) 
        )
        self.logZ = nn.Parameter(torch.tensor(3.0))

    def forward(self, state_tensor):
        return self.network(state_tensor)

    def to_input_tensor(self, state, grid_size):

        one_hot = torch.zeros(grid_size * grid_size)
        index = state[0] * grid_size + state[1]
        one_hot[index] = 1.0
        coords = torch.tensor([state[0] / (grid_size - 1), state[1] / (grid_size - 1)], dtype=torch.float32)
        return torch.cat([one_hot, coords])

grid = create_reward_grid(grid_size,high_reward_centers,high_reward,mid_reward,low_reward)
env = GridEnvironment(grid)

def meta_model_sampler(model_ensemble, env, n_samples, max_length, temperature=1.0):
    trajectories = []
    action_traj = []

    with torch.no_grad():
        for _ in range(n_samples):
            
            path = []
            actions = []
            state = env.get_initial_state()
            
            visited_states = {state}
            path.append(state)

            for _ in range(max_length):

                # any model can be used to perform the embedding
                state_tensor = model_ensemble[0].to_input_tensor(state, env.grid_size)
                valid_actions = env.get_valid_actions(state)

                unvisited_actions = []
                for action_ in valid_actions:
                    if action_ == 4:
                        unvisited_actions.append(action_)
                        continue
                    next_state = env.step(state, action_)
                    if next_state not in visited_states:
                        unvisited_actions.append(action_)
                
                if not unvisited_actions: 
                    break

                mask = torch.full((env.action_space_size,), -torch.inf, device=state_tensor.device)
                mask[unvisited_actions] = 0
                
                action_logits = torch.zeros(env.action_space_size)
                M = len(model_ensemble)
                for n in range(M):
                    action_logits += (1/M)*model_ensemble[n](state_tensor) + mask
                
                action_distribution = Categorical(logits=action_logits / temperature)
                action = action_distribution.sample().item()

                if action == 4: 
                    break
                
                state = env.step(state, action)
                visited_states.add(state)
                path.append(state)
                actions.append(action)
            
            trajectories.append(path)
            action_traj.append(actions)
            
    return trajectories, action_traj

def ensemble_flows_along_traj(model_ensemble,env,traj,actions):
    flows = []
    visited_states =[]
    M = len(model_ensemble)

    for t in range(0,len(traj)-1):
        visited_states.append(tuple(traj[t]))
        state_tensor = model_ensemble[0].to_input_tensor(traj[t], env.grid_size)
        valid_actions = env.get_valid_actions(traj[t])

        unvisited_actions = []
        for action_ in valid_actions:
            if action_ == 4:
                unvisited_actions.append(action_)
                continue
            next_state = env.step(traj[t], action_)
            if next_state not in visited_states:
                unvisited_actions.append(action_)

        mask = torch.full((env.action_space_size,), -torch.inf, device=state_tensor.device)
        mask[unvisited_actions] = 0

        flow_ensemble = np.zeros(M)
        
        for m in range(M):
            action_logits = model_ensemble[m](state_tensor) + mask
            probs = torch.softmax(action_logits, dim=-1).detach().numpy()
            flow_ensemble[m] = probs[actions[t]]
        flows.append(flow_ensemble)
    return np.array(flows)


def policy_along_traj(model,env,traj):
    dists = []
    visited_states =[]

    for t in range(0,len(traj)):
        visited_states.append(tuple(traj[t]))
        state_tensor = model.to_input_tensor(traj[t], env.grid_size)
        valid_actions = env.get_valid_actions(traj[t])

        unvisited_actions = []
        for action_ in valid_actions:
            if action_ == 4:
                unvisited_actions.append(action_)
                continue
            next_state = env.step(traj[t], action_)
            if next_state not in visited_states:
                unvisited_actions.append(action_)

        mask = torch.full((env.action_space_size,), -torch.inf, device=state_tensor.device)
        mask[unvisited_actions] = 0
        
        action_logits = model(state_tensor) + mask
        probs = torch.softmax(action_logits, dim=-1).detach().numpy()
        dists.append(probs)
    return np.array(dists)

def enesemble_policy_along_traj(model_ensemble,env,traj):
    ensemble_dists = []
    visited_states =[]
    M = len(model_ensemble)
    dists = np.zeros((M,len(traj),5))

    for t in range(0,len(traj)):
        visited_states.append(tuple(traj[t]))
        state_tensor = model_ensemble[0].to_input_tensor(traj[t], env.grid_size)
        valid_actions = env.get_valid_actions(traj[t])

        unvisited_actions = []
        for action_ in valid_actions:
            if action_ == 4:
                unvisited_actions.append(action_)
                continue
            next_state = env.step(traj[t], action_)
            if next_state not in visited_states:
                unvisited_actions.append(action_)

        mask = torch.full((env.action_space_size,), -torch.inf, device=state_tensor.device)
        mask[unvisited_actions] = 0
        
        

        for m, model in enumerate(model_ensemble):
            action_logits = model(state_tensor) + mask
            probs = torch.softmax(action_logits, dim=-1).detach().numpy()
            dists[m,t,:] = probs
    return dists