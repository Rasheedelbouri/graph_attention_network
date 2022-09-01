import csv
import numpy as np
import torch
import random

from matplotlib import pyplot as plt
from constants import TEST_IDX, DATAPATH

from deepchem.feat import MolGraphConvFeaturizer as MGCF

from utils import get_indexed_list

class dataloader:

    def __init__(self, path, batchsize):
        self.path = path
        self.batchsize = batchsize
        self.counter = 0
        self.get_tensors()
        self.indices = list(range(len(self.train_molecules)))

    def get_smiles_and_logDs(self):
        with open(self.path) as fh:
            smiles = []
            logDs = []
            header = True
            for row in csv.reader(fh):
              if header:
                header = False
                continue
              smiles.append(row[2])
              logDs.append(float(row[1]))

        return smiles, logDs

    def train_test_split(self):
        smiles, logDs = self.get_smiles_and_logDs()
        Z = 1#self.get_normalising_constant(logDs)
        train_smiles, train_logDs = [smiles[i] for i in range(len(smiles)) if i not in TEST_IDX], torch.tensor([logDs[i]/Z for i in range(len(smiles)) if i not in TEST_IDX])
        test_smiles, test_logDs = [smiles[i] for i in range(len(smiles)) if i in TEST_IDX], torch.tensor([logDs[i]/Z for i in range(len(smiles)) if i in TEST_IDX])

        return train_smiles, train_logDs, test_smiles, test_logDs

    def get_normalising_constant(self, vector):
        return max(abs(vector))


    def featurise_graph(self, smiles):
        return [MGCF(use_edges=True).featurize(smiles)]


    def get_atom_features_and_adjacency(self, feat_graphs: list):
      molecules = []
      adjacency = []
      edges = []
      edge_indices = []
      for feat_graph in feat_graphs[0]:
        molecules.append(torch.tensor(feat_graph.node_features)) #node features
        edges.append(torch.tensor(feat_graph.edge_features)) # edge features
        edge_indices.append(torch.tensor(feat_graph.edge_index)) #indices showing what's connected to what
        adj = torch.zeros(feat_graph.num_nodes, feat_graph.num_nodes)
        for x, y in zip(feat_graph.edge_index[0], feat_graph.edge_index[1]): #cycle through lists of indices: can also do this and reflect along the diagonal for undirected
          adj[x,y] = 1
        adjacency.append(adj)

      return molecules, edges, edge_indices, adjacency

    def get_tensors(self):
        train_smiles, self.train_logDs, test_smiles, self.test_logDs = self.train_test_split()
        train_graphs, test_graphs = self.featurise_graph(train_smiles), self.featurise_graph(test_smiles)
        self.train_molecules, self.train_edges, self.train_edge_indices, self.train_adj = self.get_atom_features_and_adjacency(train_graphs)
        self.test_molecules, self.test_edges, self.test_edge_indices, self.test_adj = self.get_atom_features_and_adjacency(test_graphs)

    def shuffle(self):
        def shuffle_by_index(*args):
            c = list(zip(*args))
            random.shuffle(c)
            return zip(*c)

        self.train_molecules, self.train_edges, self.train_edge_indices, self.train_adj, self.train_logDs = shuffle_by_index(self.train_molecules, self.train_edges, self.train_edge_indices, self.train_adj, self.train_logDs)


    def __iter__(self):
        self.indices = list(range(len(self.train_molecules)))
        return self

    def __len__(self):
        return len(self.train_molecules)

    def __next__(self):

        if len(self.indices) <= self.batchsize and len(self.indices) > 0:
            x = np.copy(self.indices)
        elif len(self.indices) == 0:
            raise StopIteration
        else:
            x = random.sample(self.indices, self.batchsize)

        self.indices = [ele for ele in self.indices if ele not in x]

        adj_list =  get_indexed_list(self.train_adj, x)
        lens = [len(ta) for ta in adj_list]
        lens.insert(0,0)
        return torch.vstack(get_indexed_list(self.train_molecules, x)), \
                torch.vstack(get_indexed_list(self.train_edges, x)), \
                torch.hstack(get_indexed_list(self.train_edge_indices, x)), \
                torch.block_diag(*adj_list), \
                lens, \
                torch.vstack(get_indexed_list(self.train_logDs, x))
