import torch
import torch.nn as nn


class FFN(nn.Module):
    
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        """Init FFN Block

        Args:
            d_model (int): embedding dimension of the model
            d_ffn (int): hidden dimension of the FFN
            dropout (float, optional): Probability for Dropout. Defaults to 0.0.
        """
        super(FFN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x):
        """Forward pass of two-layer MLP with GeLU activation function and dropout

        Args:
            x (Tensor): Input Tensor of dimension (...,d_model)

        Returns:
            Tensor: Same shape as input
        """
        return self.model(x)
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, p_dropout, d_k):
        """Init SDPA Layer

        Args:
            p_dropout (float): Probability for Dropout on attention scores
            d_k (int): Dimension of the keys (used for scaling with 1/sqrt(d_k))
        """
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p_dropout)
        self.sqrt_dk = d_k ** 0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        """Forward pass of SDPA.

        For the arguments shape, B denotes the batch size,
        H the number of heads,
        L the sequence length and d_k the embedding dimension of one head.

        Args:
            q (Tensor): Queries of shape (B, H, L, d_k)
            k (Tensor): Keys of shape (B, H, L, d_k)
            v (Tensor): Values of shape (B, H, L, d_k)
            mask (Tensor, optional): If provided, the mask of shape (B, 1, L, L). Positions which equal True, are masked away. Defaults to None.

        Returns:
            Tensor: Attention output of shape (B, H, L, d_k)
        """
       
        score: torch.Tensor = (q @ k.transpose(2, 3)) / self.sqrt_dk
        
        if mask is not None:
            score.masked_fill_(mask == True, torch.finfo(score.dtype).min)
            
        score = self.dropout(self.softmax(score))
        attention = score @ v
        
        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, p_dropout: float):
        """Init MHA layer

        Args:
            d_model (int): embedding dimension of the model
            heads (int): number of heads for MHA
            p_dropout (float): Probability for Dropout on the attention scores
        """
        super(MultiHeadAttention, self).__init__()
        
        self.heads = heads
        self.d_k = d_model // heads
        self.w_q = nn.Linear(d_model, self.d_k * heads, bias=False)
        self.w_k = nn.Linear(d_model, self.d_k * heads, bias=False)
        self.w_v = nn.Linear(d_model, self.d_k * heads, bias=False)
        self.w_o = nn.Linear(heads * self.d_k, d_model, bias=False)
        
        self.attention_layer = ScaledDotProductAttention(p_dropout, self.d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        """Forward pass of the MHA

        B denotes the batch size,
        L the sequence length and d_model the embedding dimension of the model.

        Args:
            q (Tensor): Queries of shape (B, L, d_model)
            k (Tensor): Keys of shape (B, L, d_model)
            v (Tensor): Values of shape (B, L, d_model)
            mask (Tensor, optional): If present, a mask of shape (B, L, L). Defaults to None.

        Returns:
            Tensor: Attention output of shape (B, L, d_model)
        """
        
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
    
    def __init__(self, d_model, num_heads, d_ffn, p_dropout, torch_attention = False):
        """Init Transformer Encoder Layer

        Args:
            d_model (int): Embedding dimension of the model
            num_heads (int): number of heads for Self Attention (MHA)
            d_ffn (int): Hidden dimension of the FFN
            p_dropout (float): Probability for Dropout
            torch_attention (bool, optional): If true, use PyTorch implementation of MHA to use FlashAttention and speedup. Defaults to False.
        """
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_att_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.torch_attention = torch_attention
        if torch_attention:
            self.self_attention = nn.MultiheadAttention(d_model, num_heads, p_dropout, batch_first=True)
        else:
            self.self_attention = MultiHeadAttention(d_model, num_heads, p_dropout)
        self.self_att_dropout = nn.Dropout(p_dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn_layer = FFN(d_model, d_ffn) 
        self.ffn_dropout = nn.Dropout(p_dropout)

        
    def forward(self, x, attn_mask=None, key_pad_mask=None):
        """Forward pass of a Transformer Encoder Layer

        Args:
            x (Tensor): Tensor of shape (B, L, d_model)
            attn_mask (Tensor, optional): If present, Mask for the attention scores of shape (L,L). Defaults to None.
            key_pad_mask (Tensor, optional): If present, mask for padding of the sequences with shape (B, L). Defaults to None.

        Returns:
            Tensor: Tensor of shape (B, L, d_mdoel)
        """
        
        # 1. layernorm
        z = self.self_att_norm(x)
        
        with torch.profiler.record_function('EncoderAttention'):
            # 2. self attention
            if self.torch_attention:
                att, _ = self.self_attention(z, z, z, attn_mask=attn_mask, key_padding_mask=key_pad_mask, need_weights=False)
            else:
                attn_mask = self._combine_att_masks(attn_mask, key_pad_mask)
                att = self.self_attention(z, z, z, mask=attn_mask)
            att = self.self_att_dropout(att)
        
        # 3. residual connection
        y = att + x
        
        with torch.profiler.record_function('EncoderFFN'):
            # 4. FFN
            out = self.ffn_layer(self.ffn_norm(y))
            out = self.ffn_dropout(out)
        out = out + y
        
        return out
    

    def _combine_att_masks(self, attn_mask, key_pad_mask):
        num_token = key_pad_mask.size(-1)
        pad = key_pad_mask.unsqueeze(1).expand(-1, num_token, num_token)
        pad = pad | pad.transpose(-2, -1)

        return attn_mask | pad
    
    
class TransformerDecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ffn, p_dropout, torch_attention = False):
        """Init Transformer Decoder Layer

        Args:
            d_model (int): Embedding dimension of the model
            num_heads (int): number of heads for Self Attention (MHA)
            d_ffn (int): Hidden dimension of the FFN
            p_dropout (float): Probability for Dropout
            torch_attention (bool, optional): If true, use PyTorch implementation of MHA to use FlashAttention and speedup. Defaults to False.
        """
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_att_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.cross_att_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.torch_attention = torch_attention
        if torch_attention:
            self.self_attention = nn.MultiheadAttention(d_model, num_heads, p_dropout, batch_first=True)
            self.cross_attention = nn.MultiheadAttention(d_model, num_heads, p_dropout, batch_first=True)
        else:
            self.self_attention = MultiHeadAttention(d_model, num_heads, p_dropout)
            self.cross_attention = MultiHeadAttention(d_model, num_heads, p_dropout)
        self.self_att_dropout = nn.Dropout(p_dropout)
        self.cross_att_dropout = nn.Dropout(p_dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn_layer = FFN(d_model, d_ffn) 
        self.ffn_dropout = nn.Dropout(p_dropout)

        
    def forward(self, x, encoder_feats, attn_mask=None, key_pad_mask=None):
        """Forward pass of a Transformer Deoder Layer

        Args:
            x (Tensor): Tensor of shape (B, L, d_model)
            encoder_feats (Tensor): Encoder features for cross attention. Tensor of shape (B, N, d_model)
            attn_mask (Tensor, optional): Mask for the self attention of the decoder Tensor of shape (L, L). Defaults to None.
            key_pad_mask (Tensor, optional): If present, mask for padding of the sequences with shape (B, L). Defaults to None.

        Returns:
            Tensor: Tensor of shape (B, L, d_model)
        """
        
        # 1. layernorm
        z = self.self_att_norm(x)
        
        with torch.profiler.record_function('DecocerSelfAttention'):
            # 2. self attention
            if self.torch_attention:
                att, _ = self.self_attention(z, z, z, attn_mask=attn_mask, key_padding_mask=key_pad_mask, need_weights=False)
            else:
                attn_mask = self._combine_att_masks(attn_mask, key_pad_mask)
                att = self.self_attention(z, z, z, mask=attn_mask)
            att = self.self_att_dropout(att)
        
        # 3. residual connection
        y = att + x
        
        z = self.cross_att_norm(y)
        with torch.profiler.record_function('DecoderCrosAttention'):
            if self.torch_attention:
                att, _ = self.cross_attention(z, encoder_feats, encoder_feats, need_weights=False)
            else:
                att = self.cross_attention(z, encoder_feats, encoder_feats)
            att = self.self_att_dropout(att)
        
        y = att + y
        
        with torch.profiler.record_function('DecoderFFN'):
            # 4. FFN
            out = self.ffn_layer(self.ffn_norm(y))
            out = self.ffn_dropout(out)
        out = out + y
        
        return out
    

    def _combine_att_masks(self, attn_mask, key_pad_mask):
        num_token = key_pad_mask.size(-1)
        pad = key_pad_mask.unsqueeze(1).expand(-1, num_token, num_token)
        pad = pad | pad.transpose(-2, -1)

        return attn_mask | pad
