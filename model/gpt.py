import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_k)

        self.out_proj = nn.Linear(n_heads * self.d_k, d_model)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)  # shape (1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2) # B, NH, T, D_K
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # dot prod of q and k
        # then compute the attention scores
        attn_scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k) # B, NH, T, T
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out

        
class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class LayerNorm(nn.Module):
    def __init__(self, n_dim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    

class AttentionBlock(nn.Module):
    def __init__ (self, config):
        super().__init__()

        self.layer_norm1 = LayerNorm(n_dim=config.d_model, bias=config.bias)
        self.attention = Attention(config.d_model, config.n_heads, config.max_seq_len)
        self.layer_norm2 = LayerNorm(n_dim=config.d_model, bias=config.bias)
        self.mlp = MLP(config.d_model, config.dropout)
    
    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)

        self.blocks = nn.ModuleList([AttentionBlock(config=config) for _ in range(config.n_layers)])

        self.layer_norm3 = LayerNorm(config.d_model, bias=config.bias)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets):
        B, T = idx.shape
        device = idx.device

        assert T <= self.config.max_seq_len, f"Cannot use seq of len {T} with max sequence len as {self.config.max_seq_len}"

        position = torch.arange(0, T, device=device)

        token_emb = self.wte(idx)
        pos_emb = self.wpe(position)

        x = token_emb + pos_emb
        

        for block in self.blocks:
            x = block(x)
        x = self.layer_norm3(x)

        if targets is not None:
            logits = self.lm_head(x)  # B, T, V
            loss = F.cross_entropy(
                input=logits.view(-1, logits.size(-1)),  # flatten B*T → rows, V → columns
                target=targets.view(-1),                 # flatten B*T → class indices
                ignore_index=-1                          # ignore padding positions (or masked)
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        idx: (B, T) tensor of indices
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self.forward(idx_cond, None)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


