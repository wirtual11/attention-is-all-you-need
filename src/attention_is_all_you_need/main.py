"""Run a deterministic, CPU-only copy-task test of the complete Transformer."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from attention_is_all_you_need.Transformer import Transformer

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 12


def make_masks(
    src: torch.Tensor, tgt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build broadcastable masks: True means a key can be attended to."""
    if src.ndim != 2 or tgt.ndim != 2 or src.size(0) != tgt.size(0):
        raise ValueError("src and tgt must be [batch, length] with matching batch sizes")
    if src.size(1) == 0 or tgt.size(1) == 0:
        raise ValueError("Source and target sequences must not be empty")
    if not src.ne(PAD_ID).any(dim=1).all() or tgt[:, 0].eq(PAD_ID).any():
        raise ValueError("Each source needs a token and each target must start with a non-PAD token")

    src_mask = src.ne(PAD_ID)[:, None, None, :]
    causal_mask = torch.ones(
        tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device
    ).tril()
    tgt_mask = tgt.ne(PAD_ID)[:, None, None, :] & causal_mask[None, None, :, :]
    return src_mask, tgt_mask, src_mask


def build_model() -> Transformer:
    return Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        max_seq_length=8,
        dropout=0.0,
    )


def make_copy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [
        [3, 4, 5],
        [5, 4, 3],
        [6, 7],
        [7, 6],
        [8],
        [9, 10, 3, 4],
        [4, 3, 10, 9],
        [3, 3, 6],
    ]
    src = pad_sequence(
        [torch.tensor(tokens, dtype=torch.long) for tokens in sequences],
        batch_first=True,
        padding_value=PAD_ID,
    )
    targets = pad_sequence(
        [torch.tensor([BOS_ID, *tokens, EOS_ID], dtype=torch.long) for tokens in sequences],
        batch_first=True,
        padding_value=PAD_ID,
    )
    return src, targets


@torch.no_grad()
def greedy_decode(model: Transformer, src: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Generate from BOS only; return tokens including EOS, but not BOS."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model.eval()
    generated = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=src.device)
    src_mask, _, _ = make_masks(src, generated)
    memory = model.encoder(src, src_mask)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_new_tokens):
        _, tgt_mask, cross_mask = make_masks(src, generated)
        log_probs = model.decoder(generated, memory, tgt_mask, cross_mask)
        next_token = log_probs[:, -1].argmax(dim=-1)
        next_token = next_token.masked_fill(finished, PAD_ID)
        generated = torch.cat((generated, next_token[:, None]), dim=1)
        finished |= next_token.eq(EOS_ID)
        if finished.all():
            break
    return generated[:, 1:]


def run_end_to_end(steps: int = 300) -> None:
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(42)
    model = build_model()
    src, targets = make_copy_batch()
    # Teacher forcing: the decoder never receives the token it must predict at this position.
    tgt_input, labels = targets[:, :-1], targets[:, 1:]
    masks = make_masks(src, tgt_input)
    criterion = nn.NLLLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    def loss_for(log_probs: torch.Tensor) -> torch.Tensor:
        return criterion(log_probs.reshape(-1, VOCAB_SIZE), labels.reshape(-1))

    model.eval()
    with torch.no_grad():
        initial_output = model(src, tgt_input, *masks)
        assert initial_output.shape == (*labels.shape, VOCAB_SIZE)
        assert torch.isfinite(initial_output).all(), "Non-finite forward output"
        torch.testing.assert_close(
            initial_output.exp().sum(dim=-1), torch.ones_like(labels, dtype=torch.float)
        )
        initial_loss = loss_for(initial_output).item()
    initial_weights = model.encoder.embedding.embedding.weight.detach().clone()
    print(f"Source: {tuple(src.shape)}, decoder input: {tuple(tgt_input.shape)}")
    print(f"Output: {tuple(initial_output.shape)} (log-probabilities)")

    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model(src, tgt_input, *masks))
        assert torch.isfinite(loss), f"Non-finite loss at step {step + 1}"
        loss.backward()
        if step == 0:
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient for {name}"
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_output = model(src, tgt_input, *masks)
        final_loss = loss_for(final_output).item()
    assert not torch.equal(initial_weights, model.encoder.embedding.embedding.weight), (
        "Optimizer did not update encoder embeddings"
    )
    print(f"NLL loss after {steps} steps: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < 0.05 and final_loss < initial_loss * 0.1, (
        "The model did not overfit the tiny batch; increase --steps"
    )

    predictions = greedy_decode(model, src, max_new_tokens=labels.size(1))
    torch.testing.assert_close(predictions, labels, rtol=0, atol=0)
    print(f"Greedy exact matches: {src.size(0)}/{src.size(0)} (including EOS and padding)")
    for source, expected, predicted in zip(src.tolist(), labels.tolist(), predictions.tolist()):
        print(f"  source={source} expected={expected} generated={predicted}")

    with TemporaryDirectory(prefix="transformer-e2e-") as directory:
        checkpoint = Path(directory) / "transformer.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = build_model()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(src, tgt_input, *masks), final_output, rtol=0, atol=0)
        torch.testing.assert_close(
            greedy_decode(restored, src, labels.size(1)), predictions, rtol=0, atol=0
        )
    print("PASS: forward, backward, learning, autoregressive generation, and checkpoint reload")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300, help="Training steps (default: 300)")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    run_end_to_end(args.steps)


if __name__ == "__main__":
    main()
