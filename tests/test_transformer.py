import io
import unittest
from contextlib import redirect_stdout

import torch
import torch.nn.functional as F

from attention_is_all_you_need.MultiHeadAttention import MultiHeadAttention
from attention_is_all_you_need.main import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    build_model,
    greedy_decode,
    make_copy_batch,
    make_masks,
    run_end_to_end,
)


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(7)

    def test_attention_matches_pytorch_with_unequal_sequence_lengths(self):
        batch_size, query_length, key_length, d_model = 2, 3, 5, 12
        query = torch.randn(batch_size, query_length, d_model)
        key = torch.randn(batch_size, key_length, d_model)
        value = torch.randn(batch_size, key_length, d_model)
        mask = torch.ones(batch_size, 1, query_length, key_length, dtype=torch.bool).tril()
        mask[1, :, :, 1] = False
        for num_heads in (1, 3, 6):
            with self.subTest(num_heads=num_heads):
                attention = MultiHeadAttention(d_model, num_heads)

                def project(layer, tokens):
                    return layer(tokens).view(
                        batch_size, -1, num_heads, d_model // num_heads
                    ).transpose(1, 2)

                expected = F.scaled_dot_product_attention(
                    project(attention.query_linear, query),
                    project(attention.key_linear, key),
                    project(attention.value_linear, value),
                    attn_mask=mask,
                    dropout_p=0.0,
                )
                expected = attention.output_linear(
                    expected.transpose(1, 2).reshape(batch_size, query_length, d_model)
                )
                actual = attention(query, key, value, mask)
                self.assertEqual(actual.shape, (batch_size, query_length, d_model))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_invalid_head_configuration(self):
        for d_model, num_heads in ((8, 3), (8, 0), (0, 2), (8, -2)):
            with self.subTest(d_model=d_model, num_heads=num_heads):
                with self.assertRaisesRegex(ValueError, "divisible"):
                    MultiHeadAttention(d_model, num_heads)

    def test_masks_use_true_for_visible_keys(self):
        src = torch.tensor([[3, 4, PAD_ID], [5, PAD_ID, PAD_ID]])
        tgt = torch.tensor([[BOS_ID, 3, 4, EOS_ID], [BOS_ID, 5, PAD_ID, PAD_ID]])
        src_mask, tgt_mask, cross_mask = make_masks(src, tgt)
        self.assertEqual(src_mask.shape, (2, 1, 1, 3))
        self.assertEqual(tgt_mask.shape, (2, 1, 4, 4))
        torch.testing.assert_close(src_mask[:, 0, 0], src.ne(PAD_ID))
        torch.testing.assert_close(cross_mask, src_mask)
        torch.testing.assert_close(tgt_mask[0, 0], torch.ones(4, 4, dtype=torch.bool).tril())
        torch.testing.assert_close(
            tgt_mask[1, 0],
            torch.tensor([
                [True, False, False, False],
                [True, True, False, False],
                [True, True, False, False],
                [True, True, False, False],
            ]),
        )

    def test_masks_reject_inputs_with_fully_masked_queries(self):
        for src, tgt in (
            (torch.zeros(1, 2, dtype=torch.long), torch.tensor([[BOS_ID]])),
            (torch.tensor([[3]]), torch.tensor([[PAD_ID, BOS_ID]])),
            (torch.empty(1, 0, dtype=torch.long), torch.tensor([[BOS_ID]])),
            (torch.tensor([[3]]), torch.empty(1, 0, dtype=torch.long)),
            (torch.tensor([3]), torch.tensor([[BOS_ID]])),
            (torch.tensor([[3], [4]]), torch.tensor([[BOS_ID]])),
        ):
            with self.subTest(src=src.tolist(), tgt=tgt.tolist()):
                with self.assertRaises(ValueError):
                    make_masks(src, tgt)

    def test_forward_returns_normalized_log_probabilities(self):
        model = build_model().eval()
        src, targets = make_copy_batch()
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size), torch.no_grad():
                source, tgt = src[:batch_size], targets[:batch_size, :-1]
                output = model(source, tgt, *make_masks(source, tgt))
                self.assertEqual(output.shape, (batch_size, tgt.size(1), VOCAB_SIZE))
                self.assertTrue(torch.isfinite(output).all())
                torch.testing.assert_close(
                    output.exp().sum(-1), torch.ones(batch_size, tgt.size(1))
                )

    def test_future_target_tokens_do_not_change_prefix_predictions(self):
        model = build_model().eval()
        src = torch.tensor([[3, 4, 5], [6, 7, PAD_ID]])
        tgt = torch.tensor([[BOS_ID, 3, 4, 5], [BOS_ID, 6, 7, EOS_ID]])
        changed = tgt.clone()
        changed[:, 2:] = torch.tensor([[9, 10], [10, 9]])
        masks = make_masks(src, tgt)
        with torch.no_grad():
            original = model(src, tgt, *masks)
            perturbed = model(src, changed, *masks)
        torch.testing.assert_close(original[:, :2], perturbed[:, :2], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(original[:, 2:], perturbed[:, 2:]))

    def test_source_padding_is_ignored_but_real_source_tokens_matter(self):
        model = build_model().eval()
        src, targets = make_copy_batch()
        tgt = targets[:, :-1]
        masks = make_masks(src, tgt)
        # Keep the original masks while perturbing hidden keys.
        changed_padding = src.masked_fill(src.eq(PAD_ID), 11)
        changed_content = src.clone()
        changed_content[:, 0] = 11
        with torch.no_grad():
            original = model(src, tgt, *masks)
            padding_output = model(changed_padding, tgt, *masks)
            content_output = model(changed_content, tgt, *masks)
        torch.testing.assert_close(original, padding_output, rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(original, content_output))

    def test_target_padding_keys_are_ignored(self):
        model = build_model().eval()
        src = torch.tensor([[3, 4], [5, 6]])
        tgt = torch.tensor([[BOS_ID, PAD_ID, 3, 4], [BOS_ID, 5, PAD_ID, EOS_ID]])
        masks = make_masks(src, tgt)
        changed = tgt.masked_fill(tgt.eq(PAD_ID), 11)
        with torch.no_grad():
            original = model(src, tgt, *masks)
            perturbed = model(src, changed, *masks)
        visible_queries = tgt.ne(PAD_ID)
        torch.testing.assert_close(
            original[visible_queries], perturbed[visible_queries], rtol=1e-5, atol=1e-6
        )

    def test_greedy_decoding_stops_at_eos_or_the_length_limit(self):
        src = torch.tensor([[3, 4], [5, PAD_ID]])
        for token, expected_length in ((EOS_ID, 1), (3, 2)):
            with self.subTest(token=token):
                model = build_model()
                with torch.no_grad():
                    model.decoder.fc.weight.zero_()
                    model.decoder.fc.bias.zero_()
                    model.decoder.fc.bias[token] = 10
                predicted = greedy_decode(model, src, max_new_tokens=2)
                torch.testing.assert_close(
                    predicted, torch.full((2, expected_length), token, dtype=torch.long)
                )

    def test_end_to_end_training_generation_and_reload(self):
        with redirect_stdout(io.StringIO()):
            run_end_to_end()


if __name__ == "__main__":
    unittest.main()
