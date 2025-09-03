from torch.distributions import Normal, Categorical
import math
import numpy as np
import torch
import random
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import torch.optim as optim
from collections import Counter
import networkx as nx
from collections import deque
from tqdm import tqdm
import pandas as pd
import random
if not hasattr(np, 'bool'):
    np.bool = bool
import chaospy as chaos
import operator
import sympy
import symb_reg_helpers as h
import os
import sys

def set_seeds(n):
    with open("random_seeds.txt", "r") as f:
        seeds = [int(line.strip()) for line in f]
    seed = seeds[n]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

device = torch.device('cpu')

target_function = lambda x: np.sin(x) + 2 - x
x_points = np.linspace(np.pi, 4*np.pi, 100)

def phi(n, t, T):
    return np.sqrt(2 / T) * np.sin((n - 0.5) * np.pi * t / T)

def lambda_n(n, T):
    return T**2 / ((n - 0.5)**2 * np.pi**2)

T = 1
N = 100
t = np.linspace(0, T, N)
M = 100  # total number of modes
n_fixed = 2  # number of deterministic modes

dist_list = [chaos.Normal(0,1), chaos.Normal(0,1)]
joint_dist = chaos.J(*dist_list)

order = 8
#nodes, weights = chaos.generate_quadrature(order, joint_dist, rule="gaussian", sparse = True)
nodes = joint_dist.sample(285,rule='sobol') # sobol space filling


def train_model(n):
    
    set_seeds(n)

    # Set the first two modes via quadrature
    Z_fixed = np.array(nodes[:,n])  # Z1, Z2

    # Sample the rest randomly
    Z_random = np.random.randn(M - n_fixed)

    Z = np.concatenate([Z_fixed, Z_random])

    W = np.zeros_like(t)

    for j in range(1, M + 1):
        W += Z[j - 1] * np.sqrt(lambda_n(j, T)) * phi(j, t, T)

    y_target = target_function(x_points) + W

    print('Training model ' + str(n))

    trained_model, _ = h.train_gflownet(y_target,x_points)

    model_name = 'symb_gfn_1_sobol_' + str(int(n+1)) + '.pth'
    with open(model_name, "wb") as f:
        print("Saving model to", model_name)
        torch.save(trained_model.state_dict(), f)
        f.flush()
        os.fsync(f.fileno())
        os.sync()
        print("Model saved", model_name)
    
    np.savetxt('Z_sobol_'+str(n+1)+'.csv',Z)
    os.sync()
    print("Wiener saved")



if __name__ == "__main__":
    # this takes in the bash parameter and trains the model with the relevant seed and quadrature node.
    train_model(int(sys.argv[1])-1)



