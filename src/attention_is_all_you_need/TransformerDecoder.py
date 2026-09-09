from torch import nn
import torch.nn.functional as F
from attention_is_all_you_need.InputEmbeddings import InputEmbeddings
from attention_is_all_you_need.positionalEncoding import PositionalEncoding
from attention_is_all_you_need.DecoderLayer import DecoderLayer

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super(TransformerDecoder, self).__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) 
                                   for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
  
    def forward(self, x, encoder_output, tgt_mask, cross_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, cross_mask)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)