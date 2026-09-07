from torch import nn
from attention_is_all_you_need.TransformerEncoder import TransformerEncoder
from attention_is_all_you_need.TransformerDecoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length)
        self.decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length)

    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, cross_mask)
        return decoder_output