import torch
import torch.nn as nn
import numpy as np
    

class FFN(nn.Module):
    
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super(FFN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.model(x)
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, p_dropout, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p_dropout)
        self.sqrt_dk = d_k ** 0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        ## input [batch_size, heads, seq_len, d_k]
        ## mask [batch_size, 0, seq_len, seq_len]
       
        score: torch.Tensor = (q @ k.transpose(2, 3)) / self.sqrt_dk
        
        if mask is not None:
            score.masked_fill_(mask == 0, -float('inf'))
            
        score = self.dropout(self.softmax(score))
        attention = score @ v
        
        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, p_dropout: float):
        super(MultiHeadAttention, self).__init__()
        
        self.heads = heads
        self.d_k = d_model // heads
        self.w_q = nn.Linear(d_model, self.d_k * heads, bias=False)
        self.w_k = nn.Linear(d_model, self.d_k * heads, bias=False)
        self.w_v = nn.Linear(d_model, self.d_k * heads, bias=False)
        self.w_o = nn.Linear(heads * self.d_k, d_model)
        
        self.attention_layer = ScaledDotProductAttention(p_dropout, self.d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # input [batch_size, seq_len, d_model]
        # mask [batch_size, seq_len, seq_len]
        
        # 1. project linearly
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 2. store heads in extra dim to compute attention parallel
        # [batch_size, seq_len, heads, d_k]
        batch_size, seq_len, _ = q.size()
        q = q.reshape(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        
        
        # 3. calculate attention
        if mask is not None:
            mask = mask.unsqueeze(1) # add dim for heads
        attention = self.attention_layer(q, k, v, mask)
        
        # 4. concatenate results
        # [batch_size, seq_len, d_k * heads]
        o = attention.transpose(1, 2).reshape(batch_size, -1, self.heads * self.d_k)
        o = self.w_o(o)
        
        return o
        

class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ffn, p_dropout):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_att_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, p_dropout)
        self.self_att_dropout = nn.Dropout(p_dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_layer = FFN(d_model, d_ffn)    

        
    def forward(self, x, att_mask=None):
        # [batch_size, seq_len, d_model]
        
        # 1. layernorm
        z = self.self_att_norm(x)
        
        # 2. self attention
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(-2)
        att = self.self_attention(z, z, z, att_mask)
        att = self.self_att_dropout(att)
        
        # 3. residual connection
        x = att + x
        
        # 4. FFN
        x = self.ffn_norm(x)
        out = self.ffn_layer(x)
        out = out + x
        
        return x
    
class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size, d_model, max_seq_len, p_dropout, learn_positions: bool = True):
        super(TokenEmbedding, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.learn_positions = learn_positions
        self.sqr_d_model = np.sqrt(d_model)
        
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p_dropout)
        
        if self.learn_positions:
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            self.register_buffer('positions', torch.arange(self.max_seq_len).unsqueeze(0))
        else:
            pe = torch.zeros((max_seq_len, d_model))
            pos = torch.arange(max_seq_len).unsqueeze(1)
            term = pos / (10000 ** (torch.arange(0, d_model, 2) / d_model))
        
            pe[:, 0::2] = torch.sin(term)
            pe[:, 1::2] = torch.cos(term)
            self.register_buffer('pe', pe)

    def forward(self, token):
        # [batch_size, seq_len]
        
        if token.size(-1) > self.max_seq_len:
            raise RuntimeError("Sequence is too long")
        
        # From "Attention Is All You Need":
        #   In our model, we share the same weight matrix between the two embedding layers 
        #   and the pre-softmax linear transformation, ... . 
        #   In the embedding layers, we multiply those weights by sqrt(d_model) .
        word_embed = self.word_embedding(token) * self.sqr_d_model
        # [batch_size, seq_len, d_model]
        
        if self.learn_positions:
            pos_embed = self.position_embedding(self.positions[:, :token.size(-1)])
        else:
            pos_embed = self.pe[:token.size(-1), :]
        
        word_embed = word_embed + pos_embed
        return self.dropout(word_embed)
