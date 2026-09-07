from torch import nn

from attention_is_all_you_need.EncoderLayer import EncoderLayer
from attention_is_all_you_need.InputEmbeddings import InputEmbeddings
from attention_is_all_you_need.positionalEncoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, src_mask):
        x = self.positional_encoding(self.embedding(x))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
