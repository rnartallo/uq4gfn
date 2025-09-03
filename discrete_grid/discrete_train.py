'''
This script will train and save N different GFN with stochastic rewards
They will then be used to create an ensemble method where uncertainty in the flows along a trajectory
can be computed
'''

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
import sys

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

class TrajectoryBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)

class PrioritizedTrajectoryBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, trajectory, reward):
        self.buffer.append(trajectory)
        self.priorities.append(reward + 1e-2) 

    def sample(self, batch_size):
        if not self.buffer:
            return []
            
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

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



def train_gflownet(grid_size, high_reward_centers, high_reward, mid_reward, low_reward, learning_rate, batch_size, buffer_capacity, episodes, max_length):
   
    reward_grid_np = create_reward_grid(grid_size, high_reward_centers, high_reward, mid_reward, low_reward)
    env = GridEnvironment(reward_grid_np)
    true_logZ = torch.log(torch.sum(env.reward_grid)).item()

    state_dim = grid_size * grid_size + 2
    gfn = GFlowNet(state_dim, env.action_space_size)
    optimizer = optim.Adam(gfn.parameters(), lr=learning_rate)
    #replay_buffer = TrajectoryBuffer(buffer_capacity)
    replay_buffer = PrioritizedTrajectoryBuffer(buffer_capacity)
    
    print("Starting GFlowNet Training...")
    pbar = tqdm(range(episodes), desc="Training Progress")
    for episode in pbar:
        trajectory = []
        state = env.get_initial_state()
        
        epsilon = max(0.1, 0.5 * (0.99 ** episode))

        for _ in range(max_length):
            if state in [t['state'] for t in trajectory]:
                break

            state_tensor = gfn.to_input_tensor(state, grid_size)
            
            with torch.no_grad():
                action_logits = gfn(state_tensor)
                valid_actions = env.get_valid_actions(state)
                mask = torch.full_like(action_logits, -torch.inf)
                mask[valid_actions] = 0
                masked_logits = (action_logits + mask) / temperature 
                action_probs = torch.softmax(masked_logits, dim=-1)

                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    action = Categorical(probs=action_probs).sample().item()

            trajectory.append({'state': state, 'action': action})
            
            if action == 4:
                break
                
            state = env.step(state, action)

        if trajectory and trajectory[-1]['action'] == 4:
            final_reward = env.get_reward(trajectory[-1]['state'])
            replay_buffer.add(trajectory, final_reward)

        if len(replay_buffer) < batch_size:
            continue

        batch = replay_buffer.sample(batch_size)
        optimizer.zero_grad()
        
        total_loss = 0
        num_sub_trajectories = 0

        for traj in batch:
            terminal_state = traj[-1]['state']
            reward = env.get_reward(terminal_state)
            log_reward = torch.log(reward + 1e-9)

            for i in range(len(traj)):
                sub_trajectory = traj[i:]
                num_sub_trajectories += 1

                fwd_log_prob = 0
                for transition in sub_trajectory:
                    s, a = transition['state'], transition['action']
                    state_tensor = gfn.to_input_tensor(s, grid_size)
                    action_logits = gfn(state_tensor)
                    valid_actions = env.get_valid_actions(s)
                    
                    mask = torch.full_like(action_logits, -torch.inf)
                    mask[valid_actions] = 0
                    masked_logits = (action_logits + mask) / temperature
                    log_probs = torch.log_softmax(masked_logits, dim=-1)
                    fwd_log_prob += log_probs[a]

                bwd_log_prob = 0
                for j in reversed(range(len(sub_trajectory) - 1)):
                    s_next = env.step(sub_trajectory[j]['state'], sub_trajectory[j]['action'])
                    action_forward = sub_trajectory[j]['action']
                    action_backward = env.inverse_action[action_forward]
                    
                    state_tensor_next = gfn.to_input_tensor(s_next, grid_size)
                    backward_logits = gfn(state_tensor_next)
                    
                    valid_actions_bwd = env.get_valid_actions(s_next)
                    mask_bwd = torch.full_like(backward_logits, -torch.inf)
                    mask_bwd[valid_actions_bwd] = 0
                    masked_logits_bwd = (backward_logits + mask_bwd) / temperature
                    log_probs_bwd = torch.log_softmax(masked_logits_bwd, dim=-1)
                    bwd_log_prob += log_probs_bwd[action_backward]

                loss = (gfn.logZ + fwd_log_prob - log_reward - bwd_log_prob)**2
                total_loss += loss

        if num_sub_trajectories > 0:
            mean_loss = total_loss / num_sub_trajectories
            mean_loss.backward()
            optimizer.step()
        pbar.set_postfix({
            "Loss": mean_loss.detach().numpy(),
            "logZ": f"{gfn.logZ.item():.4f}",
            "True logZ": f"{true_logZ:.4f}"
        })
        
    print("Training finished.")
    return gfn, reward_grid_np, env

# Set grid options

grid_size = 10
high_reward = 200
mid_reward = 40
low_reward = 0.1

p = 0.4 

N = 35 # number of models in the ensemble

# training parameters

episodes = 20000
lr = 1e-3
batch_size = 64
buffer = 10000
max_length = 2 * grid_size
temperature = 0.4

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

    print('Training GFlowNet ' + str(n))

    # sample a reward grid
    high_reward_centers = shift_reward_centers(p)

    # train the gflownet
    gfn, reward_grid, grid_env = train_gflownet(grid_size, high_reward_centers, high_reward, mid_reward, low_reward,lr, batch_size, buffer, episodes, max_length)

    # save the model
    model_name = 'test_gfn' + str(int(n)) + '.pth'
    torch.save(gfn.state_dict(), model_name)

    # save the reward grid
    grid_name = 'test_grid' + str(int(n)) + '.csv'
    np.savetxt(grid_name, reward_grid, delimiter=",")


if __name__ == "__main__":
    # this takes in the bash parameter and trains the model with the relevant seed and quadrature node.
    train_model(int(sys.argv[1])-1)


