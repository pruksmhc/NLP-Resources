"""
config = GPTTestConfig(50304, 1024)
model = GPT(config)
model.train()
"""
from torch import nn
import torch
import math
from torch.nn import functional as F

class GPTConfig:
    attn_dropout = 0.1
    embed_dropout = 0.1
    ff_dropout = 0.1
    
    def __init__(
        self, vocab_size, max_len, **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        for key, value in kwargs.items():
            setattr(self, key, value)

class GPT1Config(GPTConfig):
    num_heads = 12
    num_blocks = 12
    embed_dim = 768

class GPTTestConfig(GPTConfig):
    num_heads = 12
    num_blocks = 1
    embed_dim = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.max_len = config.max_len
        self.tok_embed = nn.Embedding(
            config.vocab_size, embed_dim
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_len, embed_dim)
        )
        self.dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_blocks)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, config.vocab_size)
    
    def forward(self, x, target=None):
        # batch_size = x.size(0)
        seq_len = x.size(1)
        assert seq_len <= self.max_len, "sequence longer than model capacity"
        
        tok_embedding = self.tok_embed(x)
        # tok_embedding.shape == (batch_size, seq_len, embed_dim)
        pos_embedding = self.pos_embed[:, :seq_len, :]
        # pos_embedding.shape == (1, seq_len, embed_dim)
        x = self.dropout(tok_embedding + pos_embedding)
        x = self.blocks(x)
        x = self.ln(x)
        x = self.fc(x)
        # x.shape == (batch_size, seq_len, vocab_size)
        loss = None
        if target is not None:
          loss = F.cross_entropy(x.view(-1, x.size(-1)), target.view(-1))
        return x[:, -1, :], loss
    
    def inference(self, x, max_new_tokens):
      # you just feed in one at a time until you have EOS.
      curr_output = x.clone()
      while curr_output.shape[1] < max_new_tokens:
        decoder_logits, _ = self.forward(curr_output, None)
        decoder_softmax = nn.Softmax(dim=-1)
        greedy_idx = torch.argmax(decoder_softmax(decoder_logits))
        greedy = torch.tensor(greedy_idx)
        curr_output = torch.cat((curr_output, greedy.unsqueeze(0).unsqueeze(0)), dim=-1)
      return curr_output

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(config.ff_dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
  

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        assert embed_dim % self.num_heads == 0, "invalid heads and embedding dimension configuration"
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.max_len, config.max_len))
            .unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # x.shape == (batch_size, seq_len, embed_dim)
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape == (batch_size, num_heads, seq_len, head_dim)
        attn = torch.matmul(q, k_t) / math.sqrt(q.size(-1))
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = torch.matmul(attn, v)
        # y.shape == (batch_size, num_heads, seq_len, head_dim)
        y = y.transpose(1, 2)
        # y.shape == (batch_size, seq_len, num_heads, head_dim)
        y = y.reshape(batch_size, seq_len, -1)
        # y.shape == (batch_size, seq_len, embed_dim)
        y = self.proj_dropout(self.proj(y))
        return y

