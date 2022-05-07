#!/usr/bin/env python
# coding: utf-8

# In[6]:


import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
from matplotlib import pyplot as plt
from torch.nn import init
from dgl.nn import GraphConv
from util import onehot
from util import generate_onehot
from distribution import log_standard_categorical
from fullGCN_model_with_feature import AuxiliaryDeepGraphGenerativeModel 
from tools import EarlyStopping


# In[7]:


# loading cora dataset
import dgl.data
dataset = dgl.data.CoraGraphDataset()
print('Number of categories:',dataset.num_classes)


# In[8]:


g = dataset[0]

features = g.ndata['feat']
lab = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
in_feats = features.shape[1]
n_classes = dataset.num_classes
n_edges = g.number_of_edges()
print("""----Data statistics------'
  #Edges %d
  #Classes %d
  #Train samples %d
  #Val samples %d
  #Test samples %d""" %
      (n_edges, n_classes,
       train_mask.int().sum().item(),
       val_mask.int().sum().item(),
       test_mask.int().sum().item()))

# add self loop
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
n_edges = g.number_of_edges()
adj = g.adj().to_dense()

# normalization
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
g.ndata['norm'] = norm.unsqueeze(1)
    
# convert y into onehot
encode = onehot(dataset.num_classes)
labels = torch.zeros(2708,dataset.num_classes)
for i in range(g.ndata['label'].size()[0]):
    labels[i] = encode(g.ndata['label'][i])


# In[9]:


y_dim = 7
z_dim = 7 # 7
a_dim = 7 # 7
h_dim = [16] # 16 

# g_dim = 2708

model = AuxiliaryDeepGraphGenerativeModel([1433, y_dim, z_dim, a_dim, h_dim]) #,g_dim
# flow = NormalizingFlows(7, n_flows=3)
# model.add_flow(flow)
model


# In[10]:


def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


# In[11]:


n1 = torch.sum(train_mask.float())
n2 = torch.sum(val_mask.float())

alpha = 2 * (n2 + n1) / n1

loss_fn1 = F.binary_cross_entropy
loss_fn2 = nn.CrossEntropyLoss()

weight_tensor, norm = compute_loss_para(adj)
weight_tensor_train = weight_tensor.reshape((2708,2708))[train_mask][:,train_mask]
weight_tensor_val = weight_tensor.reshape((2708,2708))[val_mask][:,val_mask]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

# We will need to use warm-up in order to achieve good performance.
# Over 200 calls to SVI we change the autoencoder from deterministic to stochastic.
beta = DeterministicWarmup(n = 200)


# In[12]:


early_stopping = EarlyStopping(patience=20)

for epoch in range(1, 201):
    
    ###################
    # train the model #
    ###################
    model.train()
    optimizer.zero_grad()
    
    reconstruction, reconstruction_X = model(g, features, labels)
    likelihood_adj = -norm*loss_fn1(reconstruction[train_mask][:,train_mask], adj[train_mask][:,train_mask], weight = weight_tensor_train)
    likelihood_X = -loss_fn1(reconstruction_X[train_mask], features[train_mask])
    prior = -log_standard_categorical(labels[train_mask])
    kld = -model.kl_divergence[train_mask] # kld has a numerical issue
    
    elbo = likelihood_adj + likelihood_X + prior - kld
    
    L = -torch.mean(elbo)
    
    logits = model.classify(g, features)
    
    classication_loss = loss_fn2(logits[train_mask], lab[train_mask])

    J_alpha = L + alpha * classication_loss

    J_alpha.backward()
    optimizer.step()

    total_loss = J_alpha.item()
    
    _, pred_idx = torch.max(logits[train_mask], 1)
    _, lab_idx = torch.max(labels[train_mask], 1)
    
    accuracy = torch.mean((pred_idx == lab_idx).float())
    
    if epoch % 10 == 0:
        print("Epoch: {}".format(epoch))
        print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss, accuracy))
        
    ######################
    # validate the model #
    ######################
    model.eval()
    
    likelihood_adj = -norm*loss_fn1(reconstruction[val_mask][:,val_mask], adj[val_mask][:,val_mask], weight = weight_tensor_val)
    likelihood_X = - loss_fn1(reconstruction_X[val_mask], features[val_mask])
    prior = -log_standard_categorical(labels[val_mask])
    kld = -model.kl_divergence[val_mask] # kld has a numerical issue

    elbo = likelihood_adj + likelihood_X + prior - kld
    
    L = -torch.mean(elbo)
    
    logits = model.classify(g, features)

    classication_loss = loss_fn2(logits[val_mask], lab[val_mask])

    J_alpha = L + alpha * classication_loss
    
    total_loss = J_alpha.item()

    _, pred_idx = torch.max(logits[val_mask], 1)
    _, lab_idx = torch.max(labels[val_mask], 1)
    
    accuracy = torch.mean((pred_idx == lab_idx).float())
        
    if epoch % 10 == 0:
        print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss, accuracy))
        
    early_stopping(accuracy)
    
    if early_stopping.early_stop:
        print("Early Stopping")
        break


# In[5]:


model.eval()

logits = model.classify(g, features)
_, pred_idx = torch.max(logits[test_mask], 1)
_, lab_idx = torch.max(labels[test_mask], 1)
accuracy = torch.mean((pred_idx == lab_idx).float())
print(accuracy)


# In[ ]:




