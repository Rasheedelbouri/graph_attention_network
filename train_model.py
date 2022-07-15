import numpy as np
import torch

import random

def get_model_performance(model, dataset, adjacency_matrix, labels):


  model.eval()

  outs=[]
  for x, adj, y in zip(dataset, adjacency_matrix, labels):
    outs.append(model(adj, x).detach().cpu())

  error = np.array(outs) - labels
  squared = error.T*error
  mean = sum(squared)/len(squared)

  return np.sqrt(mean)



def train_model(model,feats, adjacency, labels, max_feats, min_feats, epochs=5):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.MSELoss()

  train_loss = []
  avg_train_loss = []
  val_rmse = []

  if torch.cuda.is_available():
    criterion = criterion.cuda()
    min_feats = torch.tensor(min_feats).cuda()
    max_feats = torch.tensor(max_feats).cuda()
  else:
    min_feats = torch.tensor(min_feats)
    max_feats = torch.tensor(max_feats)

  z = list(zip(feats, adjacency, labels))
  random.shuffle(z)
  feats, adjacency, labels = zip(*z)

  for epoch in range(epochs):
      print('epoch '+str(epoch))
      for i, (x, adj, y) in enumerate(zip(feats, adjacency, labels)):
        model.train()
        if torch.cuda.is_available():
          x = x.cuda()
          adj = adj.cuda()
          y = torch.tensor(y).cuda()
        else:
          y = torch.tensor(y)

        x = (x-min_feats)/(max_feats - min_feats)
        x = x.float()
        output = model(adj, x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        train_loss.append(loss.detach().cpu())
        avg_train_loss.append(sum(train_loss)/len(train_loss))
        optimizer.step()
        print(i)
        if i%300 == 0:
            val_molecules, val_adjacency, val_logDs = zip(*random.sample(list(zip(test_molecules, test_adjacency_matrices, test_logDs)), 20))
            val_rmse.append(get_model_performance(model, val_molecules, val_adjacency, val_logDs))


  return model, avg_train_loss, val_rmse


device = 'cuda'if torch.cuda.is_available() else 'cpu'
model = GAT(feats_in_size = len(feat_names), num_attention_heads=5, num_hidden_nodes=15, device=device)
model, loss, val_rmse = train_model(model,molecules, adjacency_matrices, train_logDs, max_feats, min_feats, epochs=1)
