# Transformer: runnable end-to-end example

A small PyTorch encoder-decoder Transformer based on
[Build Your Own Transformer: A Complete Step-by-Step Implementation Guide](https://medium.com/@anjilakshetri/build-your-own-transformer-a-complete-step-by-step-implementation-guide-4680443df83b).

## Run the Python example

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked
uv run --locked attention-is-all-you-need
```

The complete Python harness is in
[`src/attention_is_all_you_need/main.py`](src/attention_is_all_you_need/main.py).
You can also run it as a module:

```bash
uv run --locked python -m attention_is_all_you_need --steps 300
```

The example uses CPU only, a fixed seed, a small two-layer model, and eight
variable-length sequences. It needs no downloaded dataset, tokenizer, or
pretrained weights; the first setup may download the locked Python dependencies.
It deliberately **overfits the training batch** to test the implementation,
not to demonstrate generalization or real translation quality.

It exercises:

1. Token IDs, embeddings, positions, encoder, decoder, and vocabulary projection,
   with different source and target lengths.
2. Source padding, target padding, causal self-attention, and cross-attention masks.
3. Shifted teacher forcing, finite gradients, and optimizer updates.
4. Learning: final loss below 0.05 and less than one tenth of the initial loss.
5. Greedy generation starting from BOS only, with exact copies of all eight
   sequences, EOS stopping, and PAD after completed sequences.
6. Saving and loading a temporary `state_dict`, with identical predictions after
   reload. The checkpoint is automatically removed.

The script raises an error on failure and prints shapes, loss, expected/generated
sequences, and a final `PASS` on success. Using too few `--steps` can fail the
learning assertions intentionally.

## Model usage

```python
import torch
from torch import nn

from attention_is_all_you_need.main import (
    PAD_ID, VOCAB_SIZE, build_model, make_copy_batch, make_masks,
)

model = build_model()
src, targets = make_copy_batch()
tgt_input = targets[:, :-1]  # BOS, then previous target tokens
labels = targets[:, 1:]     # Next tokens, including EOS

src_mask, tgt_mask, cross_mask = make_masks(src, tgt_input)
log_probs = model(src, tgt_input, src_mask, tgt_mask, cross_mask)
loss = nn.NLLLoss(ignore_index=PAD_ID)(
    log_probs.reshape(-1, VOCAB_SIZE),
    labels.reshape(-1),
)
loss.backward()
assert torch.isfinite(loss)
```

The corrected model API is
`model(src, tgt_input, src_mask, tgt_mask, cross_mask)`, not the article's
single-input API. Source and target are integer tensors shaped `[batch, length]`.
The output is **log-probabilities**, shaped `[batch, target_length, vocab_size]`,
so use `NLLLoss`, not a loss expecting ordinary probabilities.

For this example, PAD=0, BOS=1, EOS=2, and content tokens are 3 through 11. Each
source must contain at least one non-PAD token; decoder inputs start with BOS.
Masks use **True = allowed**, which matches this custom attention layer (not the
boolean-mask convention of `nn.Transformer`). Their broadcastable shapes are
`[batch, 1, 1, source_length]` for source/cross-attention and
`[batch, 1, target_length, target_length]` for decoder self-attention. The latter
combines visible non-PAD keys with a **lower-triangular** causal mask.
PAD target labels are excluded from the loss.

## Review of the article's final code

The conceptual walkthrough is useful, but its snippets do not form a runnable
encoder-decoder model without corrections:

| Issue | Required correction |
| --- | --- |
| `softmax(scores, dim=1)` normalizes over heads, not keys. Causal masking can then produce NaNs. | Normalize over the last dimension (`dim=-1`). |
| `combine_heads` reshapes to `[batch, 1, d_model]`. | Preserve the full query sequence length, including when source and target lengths differ. |
| The article's encoder layer has `__int__` instead of `__init__`. | Use `__init__`; the local encoder layer already had this correction. |
| Encoder/decoder constructors omit positional encoding's required `dropout` argument. | Pass all three arguments. |
| The enhanced decoder layer requires encoder memory and a cross mask, but the decoder wrapper still uses its old signature. | Thread encoder output and both masks through the wrapper and every decoder layer. |
| The top-level model feeds the same `x` to both encoder and decoder. | Accept distinct source and shifted target sequences; otherwise this cannot implement ordinary seq2seq training, and copying targets through the encoder leaks their future content. |
| The mask explanation calls the allowed region upper triangular; the decoder output is called probabilities. | The allowed causal region is lower triangular, and `log_softmax` returns log-probabilities. |

This repository also needed the missing `TransformerEncoder` module, imports
matching the existing `positionalEncoding.py` filename, and a working package
entry point. Those are included.

## Regression tests

The standard-library `unittest` suite requires no extra test dependencies:

```bash
uv run --locked python -m unittest discover -s tests -v
```

It compares custom attention against PyTorch's scaled-dot-product reference,
checks mask shapes and semantics, verifies that future target tokens and masked
padding cannot affect visible predictions, and confirms that real source tokens
do affect the decoder. It also covers invalid head/mask inputs, EOS/length-limited
generation, and the complete training/generation/checkpoint round trip.