import torch
from torch import nn

class TinyLM(nn.Module):
    def __init__(self, vocab_size=32000, dim=256, depth=4, heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.encoder(x)
        logits = self.lm_head(x)
        return logits, None   # no attention extraction for now
