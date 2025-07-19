import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Rudimentary MLP with SwiGLU activation function."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up  # SwiGLU
        x = self.dropout(x)
        return self.down_proj(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_head=12, bias=True, dropout=0):
        super().__init__()
        assert embed_dim % num_head == 0
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.dropout = dropout

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # (batch, time, channels)

        # Project once to get q, k, v
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_head, T, head_dim)
        q = q.view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_head, self.head_dim).transpose(1, 2)

        # Apply scaled dot-product attention with causal masking
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )  # (B, num_head, T, head_dim)

        # Recombine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection and dropout
        y = self.out_proj(attn_output)
        y = self.resid_dropout(y)

        return y


class Block(nn.Module):
    def __init__(self, embed_dim, num_head, mlp_hidden_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.RMSNorm(embed_dim)
        self.ln2 = nn.RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim=embed_dim, num_head=num_head, dropout=dropout)
        self.mlp = MLP(embed_dim, mlp_hidden_mult * embed_dim, embed_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Simplest possible LLM.

    It predicts the next character (or token) based only on the immediately
    preceding character.

    """
    def __init__(
        self,
        vocab_size,
        block_size,
        number_of_layers=12,
        embed_dim=512,
        number_of_heads=8,
        mlp_hidden_mult=4,
        dropout=0,
        max_seq_len=5000,
        weight_tying=False,
    ):
        super().__init__()
        # Lookup table for logits. Init with a random value.
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        
        # Precompute and store position indices as a buffer
        self.pos_embed_table = nn.Embedding(block_size, embed_dim)
        self.register_buffer("position_ids", torch.arange(block_size).unsqueeze(0))  # (1, T)

        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, number_of_heads, mlp_hidden_mult=mlp_hidden_mult, dropout=dropout)
                for _ in range(number_of_layers)
            ]
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size
        self.embed_dim = embed_dim

        # Weight tying from the GPT2 paper, this improves stability supposedly
        if weight_tying:
            self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx):
        """
        B: Batch size
        T: Block size
        C: Number of channels after embedding. (vocab size)

        Args:
            idx (torch.Tensor): (B, T) tensor
            targets (torch.Tensor): (B, T) tensor

        Returns:
            A tensor of size B, T, C wher
        """
        B, T = idx.shape
        # Embedding in pytorch expects a long for memory adressing.
        token_embed = self.token_embedding_table(idx.long())
        pos_embed = self.pos_embed_table(self.position_ids[:, :T])
        x = token_embed + pos_embed

        x = self.blocks(x)

        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Continues the generation in the time dimension.

        Args:
            idx (torch.Tensor): Input tensor
            max_new_tokens (int): Number of generated tokens.
            temperature (float, optional): Scale the logits by this value. Defaults to
                1.0.
            top_k (int, optional): Crop the logits to only the top k.

        Returns:
            torch.Tensor: Larger tensor of size .....
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)
            # Look at the last logit, this is the prediction for what comes next
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Now sample from the model distribution to generate a new token.
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
