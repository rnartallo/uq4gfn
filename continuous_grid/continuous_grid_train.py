''' Import libraries '''

from torch.distributions import MultivariateNormal
from scipy.stats import norm
import math
import numpy as np
import torch
import random
import chaospy as chaos
import sys
if not hasattr(np, 'bool'):
    np.bool = bool
import os

device = torch.device('cpu')


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

    @property # decoration
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

def train(batch_size, trajectory_length, env, device, n_iterations):

    forward_model, backward_model, logZ, optimizer = setup_experiment()
    losses = []
    true_logZ = env.log_partition

    for it in range(n_iterations):
        optimizer.zero_grad()

        x = initalize_state(batch_size, device, env)

        trajectory = torch.zeros((batch_size, trajectory_length + 1, 3), device=device)
        logPF = torch.zeros((batch_size,), device=device)
        logPB = torch.zeros((batch_size,), device=device)

        # Forward loop to generate trajectory and compute logPF.
        for t in range(trajectory_length):
            policy_dist = get_policy_dist(forward_model, x)  
            action = policy_dist.sample()  
            logPF += policy_dist.log_prob(action)  

            new_x = step(x, action)
            trajectory[:, t + 1, :] = new_x
            x = new_x

        # Backward loop to compute logPB
        for t in range(trajectory_length, 1, -1):
            backward_policy_dist = get_policy_dist(backward_model, trajectory[:, t, :])  
            action = trajectory[:, t - 1, :2] - trajectory[:, t, :2]  
            logPB += backward_policy_dist.log_prob(action)

        final_pos = trajectory[:, -1, :2]
        log_reward = env.log_reward(final_pos[:, 0], final_pos[:, 1])


        # traj balance loss.
        loss = (logZ + logPF - logPB - log_reward).pow(2).mean()  
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

       

    return (forward_model, backward_model, logZ)



''' Training parameters '''
n_iterations = 5000
batch_size = 256

''' Problem parameters '''
trajectory_length = 5
min_policy_std = 0.1
max_policy_std = 1.0

''' Sample reward inputs '''
noise = 0.1
n_samples = 200

dist_list = [chaos.Normal(-1,np.sqrt(noise)), chaos.Normal(-1,np.sqrt(noise)),chaos.Normal(1,np.sqrt(noise)), chaos.Normal(1,np.sqrt(noise))]
joint_dist = chaos.J(*dist_list)

order = 4
#nodes, weights = chaos.generate_quadrature(order, joint_dist, rule="gaussian", sparse = True)
nodes, weights = chaos.generate_quadrature(order , joint_dist, rule="clenshaw_curtis",sparse=True,growth=True)

no_samples = nodes.shape[1]


""" Make one large function that calls everything given n"""

def set_seeds(n):
    with open("random_seeds.txt", "r") as f:
        seeds = [int(line.strip()) for line in f]
    seed = seeds[n]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

def train_model(n):
    
    set_seeds(n)

    mu1 = nodes[:2,n]
    mu2 = nodes[2:,n]
    
    # define uncertain reward environment
    print('Training model ' + str(n))

    env = PlaneEnvironment(mus=[tuple(mu1),tuple(mu2)], variances=[0.3, 0.3], n_sd=4.5, init_value=torch.tensor([0,0]))

    forward_model, backward_model, logZ = train(batch_size, trajectory_length, env, device, n_iterations)

    model_name1 = 'cont_for_gfn_quad_cc_' + str(int(n+1)) + '.pth'
    with open(model_name1, "wb") as f:
        print("Saving forward model to", model_name1)
        torch.save(forward_model.state_dict(), f)
        f.flush()
        os.fsync(f.fileno())
        os.sync()
        print("Model saved", model_name1)

    model_name = 'cont_back_gfn_quad_cc_' + str(int(n+1)) + '.pth'
    with open(model_name, "wb") as f:
        torch.save(backward_model.state_dict(), f)
        f.flush()           # push to kernel
        os.fsync(f.fileno())
        os.sync()

    return forward_model, backward_model
    
    
if __name__ == "__main__":
    # this takes in the bash parameter and trains the model with the relevant seed and quadrature node.
    train_model(int(sys.argv[1])-1)

