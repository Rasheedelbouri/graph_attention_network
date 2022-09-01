import numpy as np
import torch
from tqdm import tqdm

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


def train_model(model, dataloader, epochs=5, batchsize=32, lr=0.001, device='cpu'):
  #set the optimiser to be ADAM
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  #set loss function as the MSE
  criterion = nn.MSELoss()

  train_loss = []
  avg_train_loss = []
  val_rmse = []

  if torch.cuda.is_available():
    criterion = criterion.to(device)


  for epoch in range(epochs):

      dataloader.shuffle()
      batch_out = []
      batch_lab = []
      print('epoch '+str(epoch))
      for i, (x, e, ei, adj, lens, y) in tqdm(enumerate(dataloader), total=int(len(dataloader)/batchsize)):
        model.train()
        if torch.cuda.is_available():
          x = x.to(device)
          adj = adj.to(device)
          y = torch.tensor(y).to(device)
        else:
          y = torch.tensor(y)

        x = x.float()
        output = model.forward(adj, x, lens)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.detach().cpu())
        avg_train_loss.append(sum(train_loss)/len(train_loss))
  return model, avg_train_loss, val_rmse



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GAT(feats_in_size = 30, num_attention_heads=[3,3], num_hidden_nodes=[40,10], device=device)
model, loss, val_rmse = train_model(model,dl, epochs=1)
