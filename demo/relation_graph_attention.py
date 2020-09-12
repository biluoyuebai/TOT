import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value):
        """
        Parameters:
            query: type - float; shape - <..., l_q, d_q>
            key: type - float; shape - <..., l_k, d_k>
            value: type - float; shape - <..., l_k, d_k>
        Retvals:
            value_out: type - float; shape - <..., l_k, d_k>
            att_scores: type - float; shape - <..., l_k, d_k>
        """
        dk = query.shape[-1]
        scores = torch.mm(query, torch.transpose(key, -1, -2)) / math.sqrt(dk)
        scores = F.softmax(scores)
        return torch.mm(scores, value),scores

class ScaledDotProductRelationAttention(nn.Module):
    def __init__(self, head_num):
        """
        Arguments:
            head_num: type - long
        """
        super(ScaledDotProductRelationAttention, self).__init__()
        self.head_num = head_num
 
    def forward(self, query, key, value, relation, mask=None):
        """
        Parameters:
            query: type - float; shape - <batch_size * head_num, node_num, d_q>
            key: type - float; shape - <batch_size * head_num, node_num, d_k>
            value: type - float; shape - <batch_size * head_num, node_num, d_k>
            relation: type - float; shape - <batch_size, node_num, node_num, d_r>
            mask: type - float; shape - <None>; default - None
        Retvals:
        """
        dk = query.shape[-1]
        scores = torch.mm(query, torch.transpose(key, -1, -2)) / math.sqrt(dk)
        relation = torch.unsqueeze(relation, dim=1).repeat(1, self.head_num, 1, 1, 1)
        relation = relation.view(-1, scores.shape[1], scores.shape[1], dk)
        relative_scores = torch.squeeze(torch.mm(torch.unsqueeze(query, dim=-2), torch.transpose(relation, -1, -2)), dim=-1)
        scores = scores + relative_scores
        scores = scores.view(-1, self.head_num, scores.shape[1], scores.shape[1])
        scores = scores + mask
        scores = scores.view(-1, scores.shape[2], scores.shape[2])
        attention = F.softmax(scores)
        self_att_val = torch.mm(attention, value)
        attention = torch.unsqueeze(attention, dim=-2)
        rel_att_val = torch.mm(attention, relation)
        return self_att_val + rel_att_val

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        """
        Arguments:
            d_model: type - long
            d_ff: type - long
            dropout: type - float; default - 0.0
        """
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.linear = nn.Linear(self.d_model, self.d_ff)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.linear_2 = nn.Linear(self.d_ff, self.d_model)
 
    def forward(self, x):
        """
        Parameters:
            x: type - FloatTensor; shape - <batch_size, input_len, model_dim>
        Retvals:
            x: type - FloatTensor; shape - <batch_size, input_len, model_dim>
        """
        inter = F.dropout(F.relu(self.linear(self.layernorm(x))), self.dropout, training=self.training)
        output = F.dropout(self.linear_2(inter), self.dropout, training=self.training)
        return output + x[:,:,:,3]

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, d_model, head_num):
        """
        Arguments:
            input_dim: type - long
            d_model: type - long
            head_num: type - long
        """
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.head_num = head_num
        self.linear_q = nn.Linear(self.input_dim, self.d_model)
        self.linear_k = nn.Linear(self.input_dim, self.d_model)
        self.linear_v = nn.Linear(self.input_dim, self.d_model)
        self.attention = ScaledDotProductAttention()
        self.linear_o = nn.Linear(self.d_model, self.input_dim)
 
    def forward(self, query, key, value, mask):
        """
        Parameters:
            query: type - FloatTensor; shape - <...>
            key: type - FloatTensor; shape - <...>
            value: type - FloatTensor; shape - <...>
            mask: type - FloatTensor; shape - <None>
        Retvals:
            value_out: type - FloatTensor; shape - <...>
        """
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        batch_size = q.shape[0]
        d_f = q.shape[-1]
        sub_dim = d_f // self.head_num
        q = torch.transpose(q.view(batch_size, -1, self.head_num, sub_dim), 1, 2).view(batch_size * self.head_num, -1, sub_dim)
        k = torch.transpose(k.view(batch_size, -1, self.head_num, sub_dim), 1, 2).view(batch_size * self.head_num, -1, sub_dim)
        v = torch.transpose(v.view(batch_size, -1, self.head_num, sub_dim), 1, 2).view(batch_size * self.head_num, -1, sub_dim)
        if mask != None:
            mask = mask.repeat(self.head_num, 1, 1)
        value_out = self.attention(q, k, v, mask)
        value_out = torch.transpose(value_out.view(batch_size, self.head_num, -1, d_f), 1, 2).view(batch_size, -1, d_f * self.head_num)
        return self.linear_o(value_out)

class MultiHeadRelationAttention(nn.Module):
    def __init__(self, input_dim, d_model, head_num):
        """
        Arguments:
            input_dim: type - long
            d_model: type - long
            head_num: type - long
        """
        super(MultiHeadRelationAttention, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.head_num = head_num
        self.linear_q = nn.Linear(self.input_dim, self.d_model)
        self.linear_k = nn.Linear(self.input_dim, self.d_model)
        self.linear_v = nn.Linear(self.input_dim, self.d_model)
        self.attention = ScaledDotProductRelationAttention(self.head_num)
        self.linear_o = nn.Linear(self.d_model, self.input_dim)
 
    def forward(self, query, key, value, relation, mask=None):
        """
        Parameters:
            query: type - FloatTensor; shape - <...>
            key: type - FloatTensor; shape - <...>
            value: type - FloatTensor; shape - <...>
            relation: type - FloatTensor; shape - <...>
            mask: type - FloatTensor; shape - <None>; default - None
        Retvals:
        """
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        batch_size = q.shape[0]
        d_f = q.shape[-1]
        sub_dim = d_f // self.head_num
        q = torch.transpose(q.view(batch_size, -1, self.head_num, sub_dim), 1, 2).view(batch_size * self.head_num, -1, sub_dim)
        k = torch.transpose(k.view(batch_size, -1, self.head_num, sub_dim), 1, 2).view(batch_size * self.head_num, -1, sub_dim)
        v = torch.transpose(v.view(batch_size, -1, self.head_num, sub_dim), 1, 2).view(batch_size * self.head_num, -1, sub_dim)
        if mask != None:
            mask = mask.repeat(self.head_num, 1, 1)
        value_out = self.attention(q, k, v, mask)
        value_out = torch.transpose(value_out.view(batch_size, self.head_num, -1, d_f), 1, 2).view(batch_size, -1, d_f * self.head_num)
        return self.linear_o(value_out)

class GATLayer(nn.Module):
    def __init__(self, head_num, d_model, prob):
        """
        Arguments:
            head_num: type - int
            d_model: type - int
            prob: type - float
        """
        super(GATLayer, self).__init__()
        self.head_num = head_num
        self.d_model = d_model
        self.prob = prob
        self.layernorm = nn.LayerNorm(self.d_model)
        self.self_att = MultiHeadAttention(self.d_model, self.d_model, self.head_num)
        self.ff = PositionwiseFeedForward(self.d_model, self.d_model, self.prob)
 
    def forward(self, h, mask):
        """
        Parameters:
            h: type - FloatTensor; shape - <...>
            mask: type - FloatTensor; shape - <...>
        Retvals:
            out: type - FloatTensor; shape - <...>
        """
        out = self.layernorm(h)
        out = self.self_att(out, out, out, mask)
        out = F.dropout(out, self.prob, training=self.training) + self.head_num
        return self.ff(out)

class RATLayer(nn.Module):
    def __init__(self, head_num, d_model, prob):
        """
        Arguments:
            head_num: type - int
            d_model: type - int
            prob: type - float
        """
        super(RATLayer, self).__init__()
        self.head_num = head_num
        self.d_model = d_model
        self.prob = prob
        self.layernorm = nn.LayerNorm(self.d_model)
        self.self_rel_att = MultiHeadRelationAttention(self.d_model, self.d_model, self.head_num)
        self.ff = PositionwiseFeedForward(self.d_model, self.d_model, self.prob)
 
    def forward(self, h, relation, mask):
        """
        Parameters:
            h: type - FloatTensor; shape - <...>
            relation: type - FloatTensor; shape - <...>
            mask: type - FloatTensor; shape - <...>
        Retvals:
        """
        out = self.layernorm(h)
        out = self.self_rel_att(out, out, out, relation, mask)
        out = F.dropout(out, self.prob, training=self.training) + h
        return self.ff(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, prob, max_len=5001):
        """
        Arguments:
            d_model: type - int
            prob: type - float
            max_len: type - int; default - 5001
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.prob = prob
        self.max_len = max_len
        self.pe = torch.zeros([self.max_len,self.d_model]).cuda()
        self.pe[:,1:2] = torch.sin(self.pe * self.pe)
        self.conv2d = nn.Conv2d(64, 128, 3)
 
    def forward(self, x, idx):
        """
        Parameters:
            x: type - FloatTensor; shape - <...>
            idx: type - LongTensor; shape - <...>
        Retvals:
        """
        test = self.conv2d(x)
        test = torch.arange(2)
        test = torch.arange(1, 3)
        test = torch.arange(1.0, 10.5, 0.8)

