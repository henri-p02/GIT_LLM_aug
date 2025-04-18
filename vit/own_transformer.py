import torch
import torch.nn as nn


class FFN(nn.Module):
    """Two layer MLP with the GeLU activation function"""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of two-layer MLP with GeLU activation function and dropout

        Args:
            x (Tensor): Input Tensor of dimension (...,d_model)

        Returns:
            Tensor: Same shape as input
        """
        return self.model(x)


class ScaledDotProductAttention(nn.Module):
    """SDPA Layer"""

    def __init__(self, p_dropout, d_k):
        """Init SDPA Layer

        Args:
            p_dropout (float): Probability for Dropout on attention scores
            d_k (int): Dimension of the keys (used for scaling with 1/sqrt(d_k))
        """
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p_dropout)
        self.sqrt_dk = d_k**0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass of SDPA.

        For the arguments shape, B denotes the batch size,
        H the number of heads,
        K the sequence length and d_k the embedding dimension of one head.

        Args:
            q (Tensor): Queries of shape (B, H, K, d_k)
            k (Tensor): Keys of shape (B, H, K, d_k)
            v (Tensor): Values of shape (B, H, K, d_k)

        Returns:
            Tensor: Attention output of shape (B, H, K, d_k)
        """

        score: torch.Tensor = (q @ k.transpose(2, 3)) / self.sqrt_dk

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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Forward pass of the MHA

        B denotes the batch size,
        L the sequence length and d_model the embedding dimension of the model.

        Args:
            q (Tensor): Queries of shape (B, L, d_model)
            k (Tensor): Keys of shape (B, L, d_model)
            v (Tensor): Values of shape (B, L, d_model)

        Returns:
            Tensor: Attention output of shape (B, L, d_model)
        """

        # 1. project linearly
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. store heads in extra dim to compute attention parallel
        # [batch_size, heads, seq_len, d_k]
        batch_size, seq_len, _ = q.size()
        q = q.reshape(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        # 3. calculate attention
        attention = self.attention_layer(q, k, v)

        # 4. concatenate results
        # [batch_size, seq_len, d_k * heads]
        o = attention.transpose(1, 2).reshape(batch_size, -1, self.heads * self.d_k)
        o = self.w_o(o)

        return o


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        p_dropout: float,
        torch_attention=False,
    ):
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

        # Use PyTorch MHA implementation for speedup through FlashAttention algorithm
        if self.torch_attention:
            self.self_attention = nn.MultiheadAttention(
                d_model, num_heads, p_dropout, batch_first=True
            )
        else:
            self.self_attention = MultiHeadAttention(d_model, num_heads, p_dropout)

        self.self_att_dropout = nn.Dropout(p_dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn_layer = FFN(d_model, d_ffn)

    def forward(self, x):
        """Forward pass of a Transormer Encoder Layer

        B denotes the batch size,
        L the sequence length and d_model the embedding dimension of the model.

        Args:
            x (Tensor): Tensor of shape (B, L, d_model)

        Returns:
            Tensor: Tensor of shape (B, L, d_model)
        """
        # [batch_size, seq_len, d_model]

        # 1. layernorm
        z = self.self_att_norm(x)

        # 2. self attention
        if self.torch_attention:
            att, _ = self.self_attention(z, z, z, need_weights=False)
        else:
            att = self.self_attention(z, z, z)
        att = self.self_att_dropout(att)

        # 3. residual connection
        y = att + x

        # 4. FFN
        out = self.ffn_layer(self.ffn_norm(y))
        out = out + y

        return out
