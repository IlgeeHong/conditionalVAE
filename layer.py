#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dgl
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


# In[2]:


### Stochastic Layer 

class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]

    to draw a sample from a distribution
    parametrised by mu and log_var.
    (q(z|x))
    """
    def reparametrize(self, mu, log_var):
        epsilon = torch.randn(mu.size())

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(0.5 * log_var)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon) # mu.addcmul=std*epsilon elementwise + mu elementwise
       
        return z
    

class FullStochastic(nn.Module):

    def reparametrize(self, mu, log_var, L):
        
        mu = torch.unsqueeze(mu,2)
        epsilon = torch.randn(mu.size())

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(0.5 * log_var)
        std = log_var.mul(0.5).exp_()
        
        L_mask = torch.tril(torch.ones(L.shape[1], L.shape[1]), diagonal=-1)
        LL = L_mask*L + torch.diag_embed(std)
        
        # z = std * epsilon + mu
        z = torch.squeeze(mu + torch.matmul(LL,epsilon))

        return z
    

class GaussianSample(Stochastic): # using two same structure layer, each have output of mu and sigma respectively
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
            
        self.mu = GraphConv(in_features, out_features, bias=False) 
        self.log_var = GraphConv(in_features, out_features, bias=False) 
              
    def forward(self, g, x):
        mu = self.mu(g, x)
        log_var = self.log_var(g, x) # F.softplus output layer = ln(1+e**x), differentiable version of ReLu

        return self.reparametrize(mu, log_var), mu, log_var # sample z using mu, log_var
    
    
class FullGaussianSample(FullStochastic): # using two same structure layer, each have output of mu and sigma respectively
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
            
        self.mu = GraphConv(in_features, out_features, bias=False) 
        self.log_var = GraphConv(in_features, out_features, bias=False)
        # L_layers = [GraphConv(in_features, out_features, bias=False) for i in range(out_features)]
        # self.L = nn.ModuleList(L_layers)
        self.L = nn.Linear(in_features, out_features*out_features, bias=False)
              
    def forward(self, g, x):
        mu = self.mu(g, x)
        log_var = self.log_var(g, x)
        L = self.L(x)
        L = L.view(x.shape[0], self.out_features, self.out_features)
        
        #T = torch.zeros(self.out_features, x.shape[0], self.out_features)
        
        #for i in range(self.out_features):
        #    for layer in self.L:
        #        y = layer(g, x)
        #        T[i] = y
        
        #L = torch.cat(torch.unbind(T), dim=1).view(x.shape[0], self.out_features, self.out_features)
        
        return self.reparametrize(mu, log_var, L), mu, log_var # sample z using mu, log_var
        
    
class GaussianSampleLinear(Stochastic): # using two same structure layer, each have output of mu and sigma respectively
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
            
        self.mu = nn.Linear(in_features, out_features) # layer for making weight matrix and bias to output mu
        self.log_var = nn.Linear(in_features, out_features)
              
    def forward(self, x):
        mu = self.mu(x) 
        log_var = F.softplus(self.log_var(x))
                                             
        return self.reparametrize(mu, log_var), mu, log_var # sample z using mu, log_var

