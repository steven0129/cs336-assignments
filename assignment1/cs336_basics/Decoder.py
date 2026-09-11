import torch
from cs336_basics.Module import LogSoftmax, StaticKVCache
from einops import rearrange

class BeamSearchDecoder:
    def __init__(self, model, eos_id=None, beam_size=5, max_length=20):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.vocab_size = model.vocab_size
        self.log_softmax = LogSoftmax(dim=-1)
        self.eos_id = eos_id

    @torch.no_grad()
    def decode(self, input_ids):
        model_param = next(self.model.parameters())
        device = model_param.device
        dtype = model_param.dtype

        prefill_len = len(input_ids)
        input_ids = torch.LongTensor(input_ids).to(device)
        repeated_input_ids = rearrange(input_ids, 'seq -> 1 seq')
        candidate_scores = torch.zeros(
            1,
            device=device,
        )

        caches = [
            StaticKVCache(1, self.max_length,
                self.model.num_heads, self.model.d_model // self.model.num_heads,
                device=device, dtype=dtype)
            for _ in self.model.layers
        ]

        logits = None
        for i in range(prefill_len):
            logits = self.model(repeated_input_ids[:, i:i + 1], caches=caches)

        for _ in range(self.max_length - prefill_len):
            next_log_probs = self.log_softmax(logits[:, -1, :])  # (beam vocab)
            total_scores = candidate_scores.unsqueeze(-1) + next_log_probs
            flat_scores = rearrange(total_scores, 'beam vocab -> (beam vocab)')
            candidate_scores, top_indices = torch.topk(flat_scores, self.beam_size, dim=-1)
            beam_indices = top_indices // self.vocab_size
            next_token_ids = top_indices % self.vocab_size

            for cache in caches:
                cache.reorder(beam_indices)

            # beam_indices also expands the prompt cache from batch=1 to batch=beam_size.
            repeated_input_ids = torch.cat([
                repeated_input_ids[beam_indices],
                next_token_ids.unsqueeze(-1)
            ], dim=-1)

            best_index = candidate_scores.argmax(dim=-1)
            best_sequence = repeated_input_ids[best_index]
            if self.eos_id is not None and best_sequence[-1] == self.eos_id:
                break
            yield best_sequence

            # KV cache already contains all previous tokens, so only feed the new token.
            logits = self.model(next_token_ids.unsqueeze(-1), caches=caches)
