from einops import rearrange, repeat
import torch
from torch import nn, Tensor

class DecoderBlock(torch.nn.Module):
  def __init__(self, dim_model: int, n_heads: int, dropout: float = 0.0):
    super().__init__()

    self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
    self.self_attn_norm = nn.LayerNorm(dim_model)
    self.ffn = nn.Sequential(
        nn.Linear(dim_model, dim_model * 4),
        nn.GELU(),
        nn.Linear(dim_model * 4, dim_model)
    )
    self.ffn_norm = nn.LayerNorm(dim_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: Tensor):
    attn_mask = torch.full(
        (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
    )
    attn_mask = torch.triu(attn_mask, diagonal=1)
    
    a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
    a1 = self.self_attn_norm (x + a1)
    a1 = self.dropout(a1)
    a2 = self.ffn(a1)
    a2 = self.ffn_norm(a1 + a2)
    a2 = self.dropout(a2)

    return a2

class Transformer(torch.nn.Module):
  def __init__(self, num_layers: int, dim_model: int, num_heads: int, num_tokens: int, seq_len: int, dropout: float = 0.0):
    super().__init__()

    self.token_embeddings = nn.Embedding(num_tokens, dim_model)
    self.position_embeddings = nn.Embedding(seq_len, dim_model)
    self.model = nn.Sequential(
        *[DecoderBlock(dim_model, num_heads, dropout=dropout) for _ in range(num_layers)],
        nn.LayerNorm(dim_model),
        nn.Linear(dim_model, num_tokens)
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs: Tensor):
    batch_size, context_len = inputs.shape

    token_embedding = self.token_embeddings(inputs)

    positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b = batch_size)
    position_embedding = self.position_embeddings(positions)

    embedding = token_embedding + position_embedding
    embedding = self.dropout(embedding)

    embedding = rearrange(embedding, 'b s d -> s b d')

    return self.model(embedding)

class MLPBlock(nn.Module):
  def __init__(self, dim_model: int, num_heads: int, dropout: float = 0.0):
    super().__init__()

    self.ffn = nn.Sequential(
      nn.Linear(dim_model, dim_model * num_heads),
      nn.ReLU(),
      nn.Linear(dim_model * num_heads, dim_model),
    )
    self.layer_norm = nn.LayerNorm(dim_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: Tensor):
    x = self.dropout(x)
    return self.layer_norm(x + self.ffn(x))

class MLP(nn.Module):
  def __init__(self, num_layers: int, dim_model: int, num_heads: int, num_tokens: int, seq_len: int, dropout: float = 0.0):
    super().__init__()

    self.token_embeddings = nn.Embedding(num_tokens, dim_model)
    self.position_embeddings = nn.Embedding(seq_len, dim_model)
    self.model = nn.Sequential(
      *[MLPBlock(dim_model * (seq_len - 1), num_heads, dropout=dropout) for _ in range(num_layers)],
      nn.LayerNorm(dim_model * (seq_len - 1)),
      nn.Linear(dim_model * (seq_len - 1), num_tokens),
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, inputs: Tensor):
    batch_size, context_len = inputs.shape

    token_embeddings = self.token_embeddings(inputs)
    position_embeddings = self.position_embeddings(torch.arange(context_len, device=inputs.device).unsqueeze(0).repeat(batch_size, 1))

    embedding = token_embeddings + position_embeddings
    embedding = self.dropout(embedding)

    embedding = rearrange(embedding, 'b s d -> b (s d)')

    return rearrange(self.model(embedding), 'm n -> 1 m n')

class LSTMModel(nn.Module):
  def __init__(self, num_layers: int, dim_model: int, hidden_dim: int, num_tokens: int, seq_len: int, dropout: float = 0.0):
    super().__init__()

    self.token_embeddings = nn.Embedding(num_tokens, dim_model)
    self.position_embeddings = nn.Embedding(seq_len, dim_model)
    self.lstm = nn.LSTM(dim_model, hidden_dim, num_layers, batch_first=True, dropout=dropout)
    self.fc = nn.Linear(hidden_dim, num_tokens)
    # self.dropout = nn.Dropout(dropout)

  def forward(self, inputs: Tensor):
    batch_size, seq_len = inputs.shape

    token_embeddings = self.token_embeddings(inputs)
    positions = repeat(torch.arange(seq_len, device=inputs.device), "p -> b p", b=batch_size)
    position_embeddings = self.position_embeddings(positions)

    embeddings = token_embeddings + position_embeddings
    # embeddings = self.dropout(embeddings)

    lstm_out, _ = self.lstm(embeddings)
    out = self.fc(lstm_out[:, -1, :])  # We only want the last output of the sequence

    return rearrange(out, 'm n -> 1 m n')
