import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import Graph

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1))
    a=q.cpu().detach().numpy()
    b=k.transpose(-2, -1).cpu().detach().numpy()
    c =scores.cpu().detach().numpy()
    scores = torch.abs(scores /  math.sqrt(d_k))
    d= scores.cpu().detach().numpy()

    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores,dim=-1)
    scores_2 = scores

    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output,scores_2


def attention_space_self(q, k, v, graph,d_k, mask=None, dropout=None):

    graph_self = graph[0].repeat(q.shape[0],q.shape[1],q.shape[2],1,1) # grpah on joints
    graph_inward = graph[1].repeat(q.shape[0],q.shape[1],q.shape[2],1,1)
    graph_outward = graph[2].repeat(q.shape[0],q.shape[1],q.shape[2],1,1)
    graph_tot = graph_self+graph_inward+graph_outward
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores /  math.sqrt(d_k)
    scores=torch.abs(scores)
    scores = scores.masked_fill(graph_tot == 0, -1e9)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores_res =torch.reshape(scores,(scores.shape[0],scores.shape[1],scores.shape[2],scores.shape[3]*scores.shape[3]))
    scores_res = F.softmax(scores_res, dim=-1)
    scores =torch.reshape(scores_res,(scores.shape[0],scores.shape[1],scores.shape[2],scores.shape[3],scores.shape[3]))
    scores_2 = scores
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output,scores_2

class MultiHeadAttention_spatial_self(nn.Module):
    def __init__(self, heads, nb_frames, dropout = 0.1):
        super().__init__()

        self.d_k = 1
        self.h = 3#heads
        self.dim=3


        self.q_linear_sp = nn.Linear(self.dim, self.dim)
        self.v_linear_sp = nn.Linear(self.dim, self.dim)
        self.k_linear_sp = nn.Linear(self.dim, self.dim)

        self.graph=torch.from_numpy(Graph.get_spatial_graph()).float().cuda()
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(45, 45)
    
    def forward(self, q, k, v,is_test, mask=None):


        bs = q.size(0)
        q= torch.reshape(q,(q.shape[0],q.shape[1],self.dim,int(q.shape[2]/self.dim)))
        q = torch.transpose(q,2,3)
        k= torch.reshape(k,(k.shape[0],k.shape[1],self.dim,int(k.shape[2]/self.dim)))
        k = torch.transpose(k,2,3)
        k = k[:,:q.shape[1],:,:]
        v= torch.reshape(v,(v.shape[0],v.shape[1],self.dim,int(v.shape[2]/self.dim)))
        v = torch.transpose(v,2,3)
        v = v[:,:q.shape[1],:,:]

        # perform linear operation and split into N heads
        k = self.k_linear_sp(k).view(bs, k.shape[1],k.shape[2], self.h, self.d_k)
        q = self.q_linear_sp(q).view(bs, q.shape[1],q.shape[2], self.h, self.d_k)
        v = self.v_linear_sp(v).view(bs, v.shape[1],v.shape[2], self.h, self.d_k)

        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(2,3)
        q = q.transpose(2,3)
        v = v.transpose(2,3)



        # calculate attention using function we will define next
        scores,att_scores = attention_space_self(q, k, v, self.graph,self.d_k, None, self.dropout)
        # concatenate heads and put through final linear layer
        concat = torch.squeeze(scores.transpose(2,3))
        concat = torch.reshape(concat,(v.shape[0],v.shape[1],v.shape[2]*v.shape[3]))

        #attention_value = np.zeros(att_scores.shape[1])
        if is_test  :
            attention_scores  = torch.squeeze(att_scores)
            attention_scores=attention_scores.cpu().detach().numpy()
            attention_value = attention_scores
        else:
            attention_value = np.zeros(att_scores.shape[1])
        output = self.out(concat)
    
        return output,attention_value


def attention_space_shared(q, k, v,distances,d_k, mask=None, dropout=None):

    scores = torch.abs(torch.matmul(q, k.transpose(-2, -1))) #added abs for test
    scores = torch.abs(scores / math.sqrt(d_k))

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores_res =torch.reshape(scores,(scores.shape[0],scores.shape[1],scores.shape[2],scores.shape[3]*scores.shape[3]))
    scores_res = F.softmax(scores_res, dim=-1)
    scores =torch.reshape(scores_res,(scores.shape[0],scores.shape[1],scores.shape[2],scores.shape[3],scores.shape[3]))
    scores_2 = scores

    scores = scores+2.0*distances.unsqueeze(2).repeat(1,1,3,1,1)
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output,scores_2#,dist_2

class MultiHeadAttention_spatial_shared(nn.Module):
    def __init__(self, heads, nb_frames, dropout = 0.1):
        super().__init__()

        self.d_k = 1
        self.h = 3#heads
        self.dim=3


        self.q_linear_sp = nn.Linear(self.dim, self.dim)
        self.v_linear_sp = nn.Linear(self.dim, self.dim)
        self.k_linear_sp = nn.Linear(self.dim, self.dim)


        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(45, 45)

    def forward(self, q, k, v,distances,is_test, mask=None):


        bs = q.size(0)
        q= torch.reshape(q,(q.shape[0],q.shape[1],self.dim,int(q.shape[2]/self.dim)))
        q = torch.transpose(q,2,3)
        k= torch.reshape(k,(k.shape[0],k.shape[1],self.dim,int(k.shape[2]/self.dim)))
        k = torch.transpose(k,2,3)
        k = k[:,:q.shape[1],:,:]
        v= torch.reshape(v,(v.shape[0],v.shape[1],self.dim,int(v.shape[2]/self.dim)))
        v = torch.transpose(v,2,3)
        v = v[:,:q.shape[1],:,:]

        # perform linear operation and split into N heads
        k = self.k_linear_sp(k).view(bs, k.shape[1],k.shape[2], self.h, self.d_k)
        q = self.q_linear_sp(q).view(bs, q.shape[1],q.shape[2], self.h, self.d_k)
        v = self.v_linear_sp(v).view(bs, v.shape[1],v.shape[2], self.h, self.d_k)


        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(2,3)
        q = q.transpose(2,3)
        v = v.transpose(2,3)



        # calculate attention using function we will define next
        scores,att_scores = attention_space_shared(q, k, v,distances,self.d_k, None, self.dropout)
        # concatenate heads and put through final linear layer
        concat = torch.squeeze(scores.transpose(2,3))
        concat = torch.reshape(concat,(v.shape[0],v.shape[1],v.shape[2]*v.shape[3]))
        if is_test  :
            attention_scores  = torch.squeeze(att_scores)
            attention_scores=attention_scores.cpu().detach().numpy()
            attention_value = attention_scores
        else:
            attention_value = np.zeros(att_scores.shape[1])
        output = self.out(concat)
        return output,attention_value
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v,is_test, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)



        # calculate attention using function we will define next
        scores,att_scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous() \
            .view(bs, -1, self.d_model)
        if is_test  :
            attention_scores  = att_scores
            attention_scores=attention_scores.cpu().detach().numpy()
            attention_value = attention_scores
        else:
            attention_value = np.zeros(att_scores.shape[1])
        output = self.out(concat)
        return output,attention_value


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
