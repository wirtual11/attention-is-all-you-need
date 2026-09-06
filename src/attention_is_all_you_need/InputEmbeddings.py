import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Multiply by sqrt(d_model) for scaling as per the original paper
        return self.embedding(x) * math.sqrt(self.d_model)