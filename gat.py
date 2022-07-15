import torch
from torch import nn

class GAT(nn.Module):

  def __init__(self, feats_in_size, num_attention_heads=5, num_hidden_nodes=10, device='cpu'):

    super().__init__()
    self.device = device

    self.num_attention_heads = num_attention_heads
    self.num_hidden_nodes = num_hidden_nodes

    #Linear layer
    self.W1 = nn.Linear(feats_in_size, num_hidden_nodes, device=self.device)
    #List of the different attention head layers
    self.attn1 = [nn.Linear(num_hidden_nodes*2, 1, device=self.device) for _ in range(num_attention_heads)]

    self.W2 = nn.Linear(num_attention_heads*num_hidden_nodes, num_hidden_nodes, device=self.device)
    self.attn2 = [nn.Linear(num_hidden_nodes*2, 1, device=self.device) for _ in range(num_attention_heads)]

    self.out = nn.Linear(num_hidden_nodes, 1, device=self.device)

    self.lrelu = nn.LeakyReLU()
    self.softmax = nn.Softmax()
    self.elu = nn.ELU()
    self.dropout = nn.Dropout(p=0.4)


  def calculate_attention(self, feats, adjacency_matrix, attn_layer):
    attentions = []
    #We cycle through the different attention heads one at a time
    for head, attn in enumerate(attn_layer):
      attn_latents_per_head=[]
      #Cycle through the atoms in the graph
      for feat in range(len(feats)):
        attention_coefs = []
        latents = []
        #This finds which atoms are bonded to which other atoms to identify the neighbourhood of the node
        links = torch.where(adjacency_matrix[feat:feat+1] == 1)
        if len(links[1]) == 0:
          #if the atom is unbonded, we don't update it's representation with an aggregation of neighbourhood representations
          attn_latents_per_head.append(feats[feat:feat+1])
          continue
        for link in links[1]:
          #Here we concatenate a neighbourhood node with the atom in question and pass through the attention layer
          attention_coefs.append(self.lrelu(attn(torch.hstack((feats[feat:feat+1], feats[link:link+1])))))
          latents.append(feats[link:link+1])
        #Once we have got the attention coefficient for each neighbour, we can now multiply the representations by the attention
        embeddings = self.softmax(torch.stack(attention_coefs)) * torch.stack(latents)
        #now we update the current atom in question with an aggregation of the attended neighbourhood nodes
        attn_latents_per_head.append(feats[feat:feat+1] + sum(embeddings))
      #capture the new node representation
      attentions.append(torch.stack(attn_latents_per_head).squeeze())

    return attentions


  def forward(self, adjacency_matrix, feats):
    #Multiply adjacency matrix to only consider neighbours then forward through linear layer. Added dropout for regularisation.
    h1 = self.dropout(self.W1(torch.inner(adjacency_matrix, feats.T)))
    #Run neighbourhood aggregation through the attention layers
    attention_h1 = self.calculate_attention(h1, adjacency_matrix, self.attn1)
    #Concatenate the various attention heads as the output of the attention layer
    stacked_attn_h1 = torch.hstack(attention_h1)
    #Activate
    h1_new = self.elu(stacked_attn_h1)

    #pass new representation through linear layer
    h2 = self.dropout(self.W2(h1_new))
    #pass representation through attention layer
    attention_h2 = self.calculate_attention(h2, adjacency_matrix, self.attn2)
    #average the attention heads this time instead of concatenating
    averaged_attn_h2 = sum(attention_h2)/self.num_attention_heads

    #activate
    h2_new = self.elu(averaged_attn_h2)
    #Aggregate node representations to capture the entire graph. We are associating a graph with a lipophilicity not individual atoms.
    avgd_h2 = sum(h2_new)/adjacency_matrix.shape[0]
    output = self.out(avgd_h2)

    return output
