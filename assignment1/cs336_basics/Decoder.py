import torch
from cs336_basics.Module import LogSoftmax
from einops import repeat, rearrange

class BeamSearchDecoder:
    def __init__(self, model, beam_size=5, max_length=20):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.vocab_size = model.vocab_size
        self.log_softmax = LogSoftmax(dim=-1)

    @torch.no_grad()
    def decode(self, input_ids):
        prefill_len = len(input_ids)
        input_ids = torch.LongTensor(input_ids)
        repeated_input_ids = rearrange(input_ids, 'seq -> 1 seq')
        candidate_scores = torch.zeros(
            1,
            device=repeated_input_ids.device,
        )

        # TODO: Deal with the case of EOS
        for _ in range(self.max_length - prefill_len):
            logits = self.model(repeated_input_ids)  # (beam seq vocab)
            next_log_probs = self.log_softmax(logits[:, -1, :])  # (beam vocab)
            total_scores = candidate_scores.unsqueeze(-1) + next_log_probs
            flat_scores = rearrange(total_scores, 'beam vocab -> (beam vocab)')
            candidate_scores, top_indices = torch.topk(flat_scores, self.beam_size, dim=-1)
            beam_indices = top_indices // self.vocab_size
            next_token_ids = top_indices % self.vocab_size
            repeated_input_ids = torch.cat([
                repeated_input_ids[beam_indices],
                next_token_ids.unsqueeze(-1)
            ], dim=-1)

            best_index = candidate_scores.argmax(dim=-1)
            best_sequence = repeated_input_ids[best_index]
            yield best_sequence