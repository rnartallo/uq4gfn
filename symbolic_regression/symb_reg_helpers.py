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
import operator
import sympy


class SymbolicRegressionEnv:

    def __init__(self, y_target, data_points, Beta = 1):
        self.tokens = ['x', '1', '2', '3', '+', '-', '*', 'sin', 'cos'] # library of terms
        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.tokens)}
        
        self.stop_action_id = len(self.tokens)
        self.action_space_size = len(self.tokens) + 1
        
        self.op_map = {'+': (operator.add, 2), '-': (operator.sub, 2),
                       '*': (operator.mul, 2), 'sin': (np.sin, 1), 'cos': (np.cos, 1),}
        self.precedence = {'+': 1, '-': 1, '*': 2, 'sin': 3, 'cos': 3}
        
        self.x_data = data_points
        self.y_target = y_target
        self.Beta = Beta
    
    def to_rpn(self, tokens):
        
        # Converts infix token list to RPN using Shunting-yard algorithm

        output_queue = deque()
        operator_stack = []
        for token in tokens:
            if token == 'x' or token.isdigit():
                output_queue.append(token)
            elif token in self.op_map:
                while (operator_stack and operator_stack[-1] in self.op_map and
                       self.precedence.get(operator_stack[-1], 0) >= self.precedence.get(token, 0)):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
        while operator_stack:
            output_queue.append(operator_stack.pop())
        return list(output_queue)

    def evaluate_rpn(self, rpn_tokens):
        # Evaluate RPN token list as a function
        stack = []
        for token in rpn_tokens:
            if token == 'x':
                stack.append(self.x_data)
            elif token.isdigit():
                stack.append(float(token))
            elif token in self.op_map:
                op_func, arity = self.op_map[token]
                if len(stack) < arity:
                    return None
                operands = [stack.pop() for _ in range(arity)]
                result = op_func(*reversed(operands))
                stack.append(result)
                
        return stack[0] if len(stack) == 1 else None

    def get_reward(self, state_tokens):
        
        # calculate reward from inverse of the MSE
        
        if not state_tokens:
            return 1e-10
        try:
            rpn = self.to_rpn(state_tokens)
            y_pred = self.evaluate_rpn(rpn)
            
            if y_pred is None or np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                mse = 1e6
            else:
                mse = np.mean((self.y_target - y_pred)**2)
        except Exception:
            mse = 1e6 #  penalize any evaluation errors

        bonus_factor_per_token = 0.2

        length_multiplier = 1.0 + (bonus_factor_per_token * len(state_tokens))

        reward = length_multiplier / (1.0 + mse)

        if mse < 1e-5:
            reward = 1000.0

        return max(reward, 1e-10)
    
    def get_MSE(self, y_pred):
        if y_pred is None or np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            mse = 1e6
        else:
            mse = np.mean((self.y_target - y_pred)**2)
        return mse



def get_action_mask(state_tokens, env):
    mask = torch.zeros(env.action_space_size)
    
    numbers_and_vars_ids = {env.token_to_id[t] for t in ['x', '1', '2', '3']}
    binary_op_ids = {env.token_to_id[t] for t in ['+', '-', '*'] if t in env.token_to_id}
    unary_op_ids = {env.token_to_id[t] for t in ['sin', 'cos'] if t in env.token_to_id}

    if not state_tokens:
        # must start with a number, variable, or unary op
        for i in numbers_and_vars_ids: mask[i] = 1
        for i in unary_op_ids: mask[i] = 1
    else:
        last_token = state_tokens[-1]
        last_token_id = env.token_to_id[last_token]
        
        if last_token_id in numbers_and_vars_ids:
            for i in binary_op_ids: mask[i] = 1
            mask[env.stop_action_id] = 1
            
        elif last_token_id in binary_op_ids:
            for i in numbers_and_vars_ids: mask[i] = 1
            for i in unary_op_ids: mask[i] = 1

        elif last_token_id in unary_op_ids:
            for i in numbers_and_vars_ids: mask[i] = 1
            
    if not state_tokens:
        mask[env.stop_action_id] = 0
        
    return mask


class GFlowNetModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64):
        super().__init__()

        # we embed each token to a latent vector
        self.embedding = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0) # +1 for padding

        # we then use an RNN with LSTM to handle the sequential structure - the final state of the LSTM is the representation of the sequence
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

        # we have three output heads
        self.logF_head = nn.Linear(hidden_dim, 1) # the log partition function
        self.Pf_head = nn.Linear(hidden_dim, vocab_size) # the forward model logits
        self.Pb_head = nn.Linear(hidden_dim, vocab_size) # the backward model logits

    def forward(self, states_padded, lengths):
        embedded = self.embedding(states_padded)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, _) = self.lstm(packed_input)
        
        hidden_state = h_n.squeeze(0)

        log_flow = self.logF_head(hidden_state).squeeze(-1)
        log_pf_logits = self.Pf_head(hidden_state)
        log_pb_logits = self.Pb_head(hidden_state) # predicts previous token

        return log_flow, log_pf_logits, log_pb_logits
    

def sample_and_evaluate(model, env, n_samples=50000, max_len = 10, temperature = 1.0, silent = False):
    # sample expressions from the trained model and show the best ones
    if not silent:
        print("Sampling expressions...")
    expressions = {}
    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            state_tokens = []
            while len(state_tokens) < max_len:
                if not state_tokens: 
                    state_ids_tensor = torch.tensor([env.stop_action_id], dtype=torch.long).unsqueeze(0)
                    lengths = torch.tensor([1])
                else:
                    state_ids_tensor = torch.tensor([env.token_to_id[t] for t in state_tokens], dtype=torch.long).unsqueeze(0)
                    lengths = torch.tensor([len(state_tokens)])
                
                _, pf_logits, _ = model(state_ids_tensor + 1, lengths)    
                # the mask for the current state
                mask = get_action_mask(state_tokens, env)
                    
                # apply mask to logits
                # set logits of invalid actions to a large negative number
                masked_logits = pf_logits.masked_fill(mask == 0, -1e9)
                pf_dist = torch.softmax(masked_logits / temperature, dim=-1)
                # sample a valid action
                action_id = torch.multinomial(pf_dist, 1).item()
                
                if action_id == env.stop_action_id:
                    break
                state_tokens.append(env.id_to_token[action_id])

            expr_str = " ".join(state_tokens)
            if not expr_str: continue 

            rpn = env.to_rpn(state_tokens)
            y_pred = env.evaluate_rpn(rpn)
            mse = env.get_MSE(y_pred)

            if expr_str not in expressions or mse < expressions[expr_str]:
                expressions[expr_str] = mse
    
    if not silent:
        print("Top 20 discovered expressions (by reward):")
    sorted_expressions = sorted(expressions.items(), key=lambda item: item[1], reverse=False)
    expr_functions =[]
    if not silent:
        for expr, mse in sorted_expressions[:20]:
            print(f"   MSE: {mse:.4f} | Expression: {expr}")
    return(sorted_expressions)


op_map_sympy = {'+': (operator.add, 2), '-': (operator.sub, 2),
                '*': (operator.mul, 2), 'sin': (sympy.sin, 1), 
                'cos': (sympy.cos, 1)}

def rpn_to_callable(rpn_tokens):
    stack = []
    x = sympy.symbols('x')

    for token in rpn_tokens:
        if token == 'x':
            stack.append(x)
        elif token.isdigit():
            stack.append(sympy.Integer(token))
        elif token in op_map_sympy:
            op_func, arity = op_map_sympy[token]
            if len(stack) < arity:
                return None
                
            operands = reversed([stack.pop() for _ in range(arity)])
            result = op_func(*operands)
            stack.append(result)

    if len(stack) != 1:
            return None

    final_expr = stack[0]
    modules = [{'sin': np.sin, 'cos': np.cos}, 'numpy']
    return sympy.lambdify([x], final_expr, modules=modules)


def flow_along_trajectory(tokens, model, env, max_len = 10, temperature = 1.5):
    flows = []
    for i , tok in enumerate(tokens):
        tokens_temp = tokens[:i]
        if not tokens_temp:
            state_ids_tensor = torch.tensor([env.stop_action_id], dtype=torch.long).unsqueeze(0)
            lengths = torch.tensor([1])
        else:
            state_ids_tensor = torch.tensor([env.token_to_id[t] for t in tokens_temp], dtype=torch.long).unsqueeze(0)
            lengths = torch.tensor([len(tokens_temp)])
        
        _, pf_logits, _ = model(state_ids_tensor + 1, lengths)    
        # the mask for the current state
        mask = get_action_mask(tokens_temp, env)
                        
        # apply mask to logits
        # set logits of invalid actions to a large negative number
        masked_logits = pf_logits.masked_fill(mask == 0, -1e9)
        pf_dist = torch.softmax(masked_logits / temperature, dim=-1).detach().numpy()

        flows.append(pf_dist[0][env.token_to_id[tok]])
    return flows

def policy_along_trajectory(tokens, model, env, max_len = 10, temperature = 1.5):
    flows = []
    for i , tok in enumerate(tokens):
        tokens_temp = tokens[:i]
        if not tokens_temp: 
            state_ids_tensor = torch.tensor([env.stop_action_id], dtype=torch.long).unsqueeze(0)
            lengths = torch.tensor([1])
        else:
            state_ids_tensor = torch.tensor([env.token_to_id[t] for t in tokens_temp], dtype=torch.long).unsqueeze(0)
            lengths = torch.tensor([len(tokens_temp)])
        
        _, pf_logits, _ = model(state_ids_tensor + 1, lengths)    
        # the mask for the current state
        mask = get_action_mask(tokens_temp, env)
                        
        # apply mask to logits
        # set logits of invalid actions to a large negative number
        masked_logits = pf_logits.masked_fill(mask == 0, -1e9)
        pf_dist = torch.softmax(masked_logits / temperature, dim=-1).detach().numpy()

        flows.append(pf_dist[0])
    return flows

def train_gflownet(y_target,x_points):

    n_episodes = 10000
    batch_size = 64
    learning_rate = 1e-3
    max_len = 10
    
    env = SymbolicRegressionEnv(y_target, x_points)
    model = GFlowNetModel(vocab_size=env.action_space_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #initial_beta = 1.0
    #final_beta = 5
    temperature = 1.5
    
    print("Starting GFlowNet Training...")

    for episode in range(n_episodes):
        batch_trajectories = []
        batch_log_rewards = []

        # generate batch of trajs
        for _ in range(batch_size):
            trajectory = []
            state_tokens = [] 
            
            while len(state_tokens) < max_len:
                if not state_tokens: 
                    state_ids_tensor = torch.tensor([env.stop_action_id], dtype=torch.long).unsqueeze(0)
                    lengths = torch.tensor([1])
                else:
                    state_ids_tensor = torch.tensor([env.token_to_id[t] for t in state_tokens], dtype=torch.long).unsqueeze(0)
                    lengths = torch.tensor([len(state_tokens)])
                
                _, pf_logits, _ = model(state_ids_tensor + 1, lengths)    
                # the mask for the current state
                mask = get_action_mask(state_tokens, env)
                    
                # apply mask to logits
                # set logits of invalid actions to a large negative number
                masked_logits = pf_logits.masked_fill(mask == 0, -1e9)
                pf_dist = torch.softmax(masked_logits / temperature, dim=-1) 
                # sample a valid action
                action_id = torch.multinomial(pf_dist, 1).item()

                prev_state_tokens = list(state_tokens)
                if action_id == env.stop_action_id:
                    trajectory.append((prev_state_tokens, action_id))
                    break
                
                state_tokens.append(env.id_to_token[action_id])
                trajectory.append((prev_state_tokens, action_id))
            
            batch_trajectories.append(trajectory)
            batch_log_rewards.append(torch.log(torch.tensor(env.get_reward(state_tokens))))

        # compute loss for the batch
        optimizer.zero_grad()
        total_loss = 0

        for i in range(batch_size):
            trajectory = batch_trajectories[i]
            if not trajectory: continue # Skip empty trajectories
            log_reward = batch_log_rewards[i]
            
            states = [item[0] for item in trajectory] + [trajectory[-1][0] + ([env.id_to_token[trajectory[-1][1]]] if trajectory[-1][1] != env.stop_action_id else [])]
            actions = [item[1] for item in trajectory]

            lengths = torch.tensor([len(s) if s else 1 for s in states])
            
            padded_states = nn.utils.rnn.pad_sequence(
                [torch.tensor([env.token_to_id[t] for t in s] or [env.stop_action_id], dtype=torch.long) + 1 for s in states],
                batch_first=True, padding_value=0
            )

            log_flows, log_pf_logits, log_pb_logits = model(padded_states, lengths)
            log_pf_dists = torch.log_softmax(log_pf_logits, dim=-1)
            log_pb_dists = torch.log_softmax(log_pb_logits, dim=-1)
            
            log_pf_sum = sum(log_pf_dists[j, actions[j]] for j in range(len(actions)))
            log_pb_sum = sum(log_pb_dists[j + 1, actions[j]] for j in range(len(actions)))
            log_z = log_flows[0]

            # Trajectory Balance Loss
            loss = (log_z + log_pf_sum - log_reward - log_pb_sum)**2
            total_loss += loss
            #if i == batch_size -1:
                #print(trajectory)

        if batch_size > 0:
            mean_loss = total_loss / batch_size
            mean_loss.backward()
            optimizer.step()
        
            if episode % 200 == 0:
                print(f"Episode {episode:5d} | Loss: {mean_loss.item():.4f} | Avg Reward: {torch.exp(torch.mean(torch.stack(batch_log_rewards))):.4f}")

    print("Training complete.")
    return model, env

def phi(n, t, T):
    return np.sqrt(2 / T) * np.sin((n - 0.5) * np.pi * t / T)

def lambda_n(n, T):
    return T**2 / ((n - 0.5)**2 * np.pi**2)


def wiener_reconstruct(Z):
    T = 1
    N = 100
    t = np.linspace(0, T, N)
    M = 100  

    W = np.zeros_like(t)
    for n in range(1, M + 1):
        W += Z[n - 1] * np.sqrt(lambda_n(n, T)) * phi(n, t, T)
    return W