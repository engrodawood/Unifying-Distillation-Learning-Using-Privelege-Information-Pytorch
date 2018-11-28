# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:22:11 2018

@author: Muhammad Dawood
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from sklearn.preprocessing import StandardScaler


# experiment 1: noiseless labels as privileged info
def synthetic_01(a,n):
    x  = np.random.randn(n,a.size)
    e  = (np.random.randn(n))[:,np.newaxis]
    xs = np.dot(x,a)[:,np.newaxis]
    y  = ((xs+e) > 0).ravel()
    return (xs,x,y)

# experiment 2: noiseless inputs as privileged info (violates causal assump)
def synthetic_02(a,n):
    x  = np.random.randn(n,a.size)
    e  = np.random.randn(n,a.size)
    y  = (np.dot(x,a) > 0).ravel()
    xs = np.copy(x)
    x  = x+e
    return (xs,x,y)

# experiment 3: relevant inputs as privileged info
def synthetic_03(a,n):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    xs = xs[:,0:3]
    a  = a[0:3]
    y  = (np.dot(xs,a) > 0).ravel()
    return (xs,x,y)

# experiment 4: sample dependent relevant inputs as privileged info
def synthetic_04(a,n):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    #xs = np.sort(xs,axis=1)[:,::-1][:,0:3]
    xs = xs[:,np.random.permutation(a.size)[0:3]]
    a  = a[0:3]
    tt = np.median(np.dot(xs,a))
    y  = (np.dot(xs,a) > tt).ravel()
    return (xs,x,y)

    
def softmax(w, t = 1.0):
    e = np.exp(w / t)
    return e/np.sum(e,1)[:,np.newaxis]

class Net(nn.Module):
    def __init__(self,d,q):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(d,2*q)
        self.hidden2 = nn.Linear(2*q,q)

    def forward(self,x):
        x = F.tanh(self.hidden1(x))
        x = F.tanh(self.hidden2(x))
        x1 = F.softmax(x)
        return x1,x    


def fitModel(model,optimizer,criterion,epochs,x,target):
    for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing x to the model
        y_pred,_ = model(x)
        # Compute and print loss
        loss = criterion(y_pred, target)
        #print(epoch, loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    return model
def do_exp(x_tr,xs_tr,y_tr,x_te,xs_te,y_te):
    t = 1
    l = 1
    l_r=0.001
    epochs=1000
    criterion = torch.nn.MSELoss()
    # scale stuff
    s_x   = StandardScaler().fit(x_tr)
    s_s   = StandardScaler().fit(xs_tr)
    x_tr  = s_x.transform(x_tr)
    x_te  = s_x.transform(x_te)
    xs_tr = s_s.transform(xs_tr)
    xs_te = s_s.transform(xs_te)
    y_tr  = y_tr*1.0
    y_te  = y_te*1.0
    y_tr  = np.vstack((y_tr==1,y_tr==0)).T
    y_te  = np.vstack((y_te==1,y_te==0)).T
    """
    Training of privilage space model
    """
    xs_tr = Variable(torch.from_numpy(xs_tr)).type(torch.FloatTensor)
    y_tr = Variable(torch.from_numpy(y_tr*1.0)).type(torch.FloatTensor)
    mlp_priv = Net(xs_tr.shape[1],2)
    optimizer = optim.RMSprop(mlp_priv.parameters(),lr=l_r)
    mlp_priv=fitModel(mlp_priv,optimizer,criterion,epochs,xs_tr,y_tr)    
    xs_te = Variable(torch.from_numpy(xs_te)).type(torch.FloatTensor)
    _,soften=mlp_priv(xs_tr)
    output,_=mlp_priv(xs_te)
    pred = torch.argmax(output,dim=1)
    pred=pred.numpy()
    res_priv=np.mean(pred==np.argmax(y_te,1))        
    
    """
    Training of regular MLP
    """
    x_tr = Variable(torch.from_numpy(x_tr)).type(torch.FloatTensor)
    mlp_reg = Net(x_tr.shape[1],2)
    optimizer = optim.RMSprop(mlp_reg.parameters(),lr=l_r)
    mlp_reg=fitModel(mlp_reg,optimizer,criterion,epochs,x_tr,y_tr)
    x_te = Variable(torch.from_numpy(x_te)).type(torch.FloatTensor)
    output,_=mlp_reg(x_te)
    pred = torch.argmax(output,dim=1)
    pred=pred.numpy()
    res_reg=np.mean(pred==np.argmax(y_te,1))

    softened=soften.detach()
    softened=softened.numpy()
    p_tr=softmax(softened,t)
    p_tr=Variable(torch.from_numpy(p_tr)).type(torch.FloatTensor)
    
    ### freezing layers
    for param in mlp_priv.parameters():
        param.requires_grad =False
    
    """
    LUPI Combination of two model
    """
    mlp_dist = Net(x_tr.shape[1],2)
    optimizer = optim.RMSprop(mlp_dist.parameters(),lr=l_r)
    criterion = torch.nn.MSELoss()
    # Training loop
    for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing x to the model
        y_pred,_ = mlp_dist(x_tr)
        # Compute and print loss
        loss1 = (1-l)*criterion(y_pred, y_tr)
        loss2=t*t*l*criterion(y_pred, p_tr)
        loss=loss1+loss2
        #print(epoch, loss.data[0])
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    output,_=mlp_dist(x_te)
    pred = torch.argmax(output,dim=1)
    pred=pred.numpy()
    res_dis=np.mean(pred==np.argmax(y_te,1))
    return np.array([res_priv,res_reg,res_dis])


# experiment hyper-parameters
d      = 50
n_tr   = 200
n_te   = 1000
n_epochs = 10
eid    = 0

np.random.seed(0)

# do all four experiments
print("\nDistillation Pytorch Classification loss function")
for experiment in (synthetic_01,synthetic_02,synthetic_03,synthetic_04):
    eid += 1
    R = np.zeros((n_epochs,3))
    for rep in range(n_epochs):
        a   = np.random.randn(d)
        (xs_tr,x_tr,y_tr) = experiment(a,n=n_tr)
        (xs_te,x_te,y_te) = experiment(a,n=n_te)
        R[rep,:] += do_exp(x_tr,xs_tr,y_tr,x_te,xs_te,y_te)
    means = R.mean(axis=0).round(2)
    stds  = R.std(axis=0).round(2)
    print (str(eid)+\
          ' # '+str(means[0])+'+/-'+str(stds[0])+\
          ' # '+str(means[1])+'+/-'+str(stds[1])+\
          ' # '+str(means[2])+'+/-'+str(stds[2]))