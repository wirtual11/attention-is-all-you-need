import torch

from attention_is_all_you_need.FeedForwardSubLayer import FeedForwardSubLayer

from .InputEmbeddings import InputEmbeddings
from .PositionalEncoding import PositionalEncoding  
from .MultiHeadAttention import MultiHeadAttention

def main() -> None:
    print("Hello from attention-is-all-you-need!")
    vocab_size = 10000  # Assuming we have 10,000 unique tokens
    d_model = 512       # 512-dimensional embeddings

    embedding_layer: InputEmbeddings = InputEmbeddings(vocab_size, d_model)

    # Sample input: batch of token IDs
    input_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # Shape: (batch_size=2, seq_len=4)
    embedded = embedding_layer(input_tokens)

    print(f"Input shape: {input_tokens.shape}")
    print(f"Embedded shape: {embedded.shape}")

    # Create positional encoding
    seq_len = 100
    dropout = 0.1
    pos_encoding = PositionalEncoding(d_model, seq_len, dropout)

    # Test with our embedded tokens
    positioned_embeddings = pos_encoding(embedded)
    print(f"Shape after positional encoding: {positioned_embeddings.shape}")

    # Define attention parameters
    d_model = 512
    num_heads = 8

    # Instantiate a MultiHeadAttention instance 
    multihead_attn = MultiHeadAttention(d_model, num_heads)

    # Pass the query, key, and value matrices through the mechanism
    output = multihead_attn(d_model, key, value)
    print(multihead_attn.shape)

    feed_forward = FeedForwardSubLayer(512, 2048)
    output = feed_forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
if __name__ == "__main__":
    main()
