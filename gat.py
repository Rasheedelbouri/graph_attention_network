from dataloading import dataloader
from constants import DATAPATH

dl = dataloader(DATAPATH, batchsize=32)

import torch
from torch import nn
import numpy as np


class GraphAttentionLayer(nn.Module):

    def __init__(self, feats_in_size: int, num_attention_heads: int, num_hidden_nodes: int, device, mean=None):

        super().__init__()
        self.device = device

        self.num_attention_heads = num_attention_heads
        self.num_hidden_nodes = num_hidden_nodes

        self.W = nn.Linear(feats_in_size, num_hidden_nodes, device=self.device)
        self.attn = nn.ModuleList([nn.Linear(num_hidden_nodes*2, 1, device=self.device) for _ in range(num_attention_heads)])

        self.mean = mean

        self.lrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=0)

    def adjacency_indices(self, adjacency_matrix):
        return (adjacency_matrix == 1).nonzero(as_tuple=False)

    def forward(self, feats, adjacency_matrix, adj_ind=None):

      if adj_ind is None:
            indices = adjacency_matrix.nonzero()
      else:
            indices = adj_ind.T

      h_feats = self.W(feats)
      feat_heads = [torch.clone(h_feats) for _ in range(self.num_attention_heads)]
      neighbours = torch.hstack((h_feats[indices[:,0].long()], h_feats[indices[:,1].long()]))
      attentions = [self.lrelu(attn(neighbours)) for attn in self.attn]

      for index in torch.unique(indices[:,0]):
          for i, attn in enumerate(attentions):
              index_list = (indices[:,0] == index).nonzero()
              feat_heads[i][index] += torch.sum(self.softmax(attn[index_list].flatten())[:,None]*feat_heads[i][indices[index_list]][:,:,1].squeeze(), dim=0).flatten()

      if self.mean:
          return torch.mean(torch.stack(feat_heads), dim=0)
      return torch.hstack(feat_heads)


class GAT(nn.Module):

  def __init__(self, feats_in_size: int, num_attention_heads: list, num_hidden_nodes: list, dropout=0.3, device='cpu'):

    super().__init__()
    assert len(num_attention_heads) == len(num_hidden_nodes)

    self.device = device
    self.num_attention_heads = num_attention_heads
    self.num_hidden_nodes = num_hidden_nodes
    self.feats_in_size = [int(feats_in_size)]
    for nh, nu in zip(self.num_attention_heads[:-1], self.num_hidden_nodes[:-1]):
        self.feats_in_size.append(int(nh*nu))

    #Linear layer
    self.gats = nn.ModuleList([GraphAttentionLayer(fs, nh, nu, self.device) for fs, nh, nu in zip(self.feats_in_size, self.num_attention_heads, self.num_hidden_nodes)])
    self.gats.append(GraphAttentionLayer(self.num_hidden_nodes[-1]*self.num_attention_heads[-1], 1, self.num_hidden_nodes[-1], self.device, mean=True))

    self.out = nn.Linear(self.num_hidden_nodes[-1], 1, device=self.device)

    self.elu = nn.ELU()
    self.layernorm = nn.LayerNorm()
    self.dropout = nn.Dropout(p=dropout)


  def forward(self, adjacency_matrix, feats, lens, adj_ind=None):
    f, adj = feats, adjacency_matrix
    for gat in self.gats:
        h = self.elu(self.dropout(gat.forward(f, adj, adj_ind)))

    cumsums = np.cumsum(lens)
    feats = self.out(f)

    graphs = torch.stack([torch.mean(feats[cumsums[c-1]:cumsums[c]]) for c in range(1,len(cumsums))])

    return graphs
