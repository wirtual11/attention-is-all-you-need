from attention_is_all_you_need import InputEmbeddings

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
