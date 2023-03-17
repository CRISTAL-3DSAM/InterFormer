import torch
import torch.nn as nn
from Sublayers_space_multihead import FeedForward, MultiHeadAttention,MultiHeadAttention_spatial_self,MultiHeadAttention_spatial_shared, Norm


#multi_head on spatial attention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_space_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_space = MultiHeadAttention_spatial_self(2, 48, dropout=dropout)# a revoir
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_space_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask,is_test):
        x2 = self.norm_1(x)
        x_val_1,att_val_1 = self.attn(x2,x2,x2,is_test,mask)
        x3 = x + self.dropout_1(x_val_1)
        x_val_2,att_val_2 = self.attn_space(x2,x2,x2,is_test,mask)
        x_space = x + self.dropout_space_1(x_val_2)
        x2 = self.norm_2(x3)
        x_space2 = self.norm_space_1(x_space)
        x_tot_2 = torch.add(x3,x_space)
        x_tot = torch.add(x2,x_space2)
        x = x_tot_2 + self.dropout_2(self.ff(x_tot))
        att_val = {'temporal_attention': att_val_1, 'spatial_attention': att_val_2}
        return x, att_val
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.norm_space_1 = Norm(d_model)
        self.norm_space_2 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.space_dropout_1 = nn.Dropout(dropout)
        self.space_dropout_2 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_space_1 = MultiHeadAttention_spatial_self(2, 48, dropout=dropout)
        self.attn_space_2 = MultiHeadAttention_spatial_shared(2, 48, dropout=dropout)

        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs,distances, src_mask, trg_mask,is_test):
        x2 = self.norm_1(x)
        x_val_1,att_val_1 = self.attn_1(x2, x2, x2,is_test, trg_mask)
        x3 = x + self.dropout_1(x_val_1)
        x_val_space_1,att_val_space_1 = self.attn_space_1(x2,x2,x2,is_test,trg_mask)
        x_space = x + self.space_dropout_1(x_val_space_1)
        x2 = self.norm_2(x3)
        x_space2 = self.norm_space_1(x_space)

        #newarch
        x_val_2,att_val_2=self.attn_2(x2, e_outputs, e_outputs, is_test, src_mask)
        x3 = x2 + self.dropout_2(x_val_2)
        x_val_space_2,att_val_space_2=self.attn_space_2(x_space2, e_outputs, e_outputs,distances, is_test, src_mask)
        x_space = x_space2 + self.space_dropout_2(x_val_space_2)
        x2 = self.norm_3(x3)
        x_space2 = self.norm_space_2(x_space)
        x_tot = torch.add(x2,x_space2)
        x = x_tot + self.dropout_3(self.ff(x_tot))


        att_val = {'temporal_attention_self': att_val_1, 'spatial_attention_self': att_val_space_1, 'temporal_attention_shared':att_val_2, 'spatial_attention_shared':att_val_space_2}
        return x,att_val