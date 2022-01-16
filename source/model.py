import math
import random
from collections import namedtuple, deque

import numpy as np
import torch
from torch import nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def sin_positional_encoding(pos_array: torch.Tensor, d_model: int = 500):
    # pos_array: [t_1,...,t_N]
    div_term = np.exp(np.arange(0, d_model / 2) * (-math.log(1000.0) / d_model)).reshape(-1, 1)
    div_term_expanded = np.zeros(div_term.shape[0] * 2).reshape(-1, 1)
    div_term_expanded[::2] = div_term
    div_term_expanded[1::2] = div_term

    pos_array = pos_array.reshape(-1, 1)
    pe = pos_array @ div_term_expanded.T  # (N, d)
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return pe.to(torch.float32)  # (N, d)


def one_hot_encoding(pos_array: torch.Tensor, d_model: int = 500):
    pe = torch.zeros((pos_array.shape[0], d_model))
    pe[np.arange(pos_array.shape[0]), pos_array] = 1
    return pe


class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Attention(nn.Module):
    def __init__(self, seq_len: int, embedding_dim: int, dropout: float = 0.4, num_head=1):
        super().__init__()
        self.seq_len = seq_len
        self.dk = embedding_dim

        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)

        self.norm_1 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_head, batch_first=True)

    def forward(self, input_tensor: torch.Tensor):
        # size (B,N,D) or (N,D)
        input_tensor = self.norm_1(input_tensor).float()
        q = self.q_linear(input_tensor)  # (B, N, D)
        k = self.k_linear(input_tensor)
        v = self.v_linear(input_tensor)
        attn_output, attn_output_weights = self.multihead_attn(q, k, v)
        output_tensor = self.norm_2(input_tensor + self.dropout(attn_output))
        return output_tensor


class ActionInModel(nn.Module):
    def __init__(self, seq_len: int, embedding_dim: int, num_head=1):
        super(ActionInModel, self).__init__()
        self.n_token = int((seq_len - 1) / 2)
        self.attn_head = Attention(seq_len + 1, embedding_dim, num_head=num_head)
        self.linear = nn.Sequential(nn.Linear((seq_len + 1) * embedding_dim, 128),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(128, 1))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # input_tensor: batch update (B,N,D); single sample: (1,N,D)
        # output_tensor: batch update (B, num_actions); single sample: (1, num_actions)
        # B: batch size, N: num cards, D: embedding dim
        action_space = input_tensor[..., self.n_token:-1, :]
        qvals = []
        for i in range(self.n_token):
            action_tensor = action_space[..., i, :].unsqueeze(-2)
            input_cat = torch.cat((input_tensor.clone().detach().requires_grad_(True),
                                   action_tensor.clone().detach().requires_grad_(True)), -2)  # concat input to action; (B,N+1,D)
            S = self.attn_head(input_cat)
            X = torch.flatten(S, start_dim=-2).float()
            qval = self.linear(X)
            qvals.append(qval)
        # qvals will be a list of num_actions
        # if batch update, each element will be a tensor of size (B, 1)
        # if single element, each element will be of size (1, 1)
        output = torch.cat(qvals, 1)
        return output


class DQNModel(nn.Module):
    # input_tensor: batch update (B,N,D); single sample: (N,D)
    # output tensor: (B, num_actions)
    def __init__(self, seq_len: int, embedding_dim: int):
        super(DQNModel, self).__init__()
        self.n_token = int((seq_len - 1) / 2)
        self.model = nn.Sequential(
            nn.Linear(embedding_dim*seq_len, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.n_token)
        )

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        return self.model(x)
