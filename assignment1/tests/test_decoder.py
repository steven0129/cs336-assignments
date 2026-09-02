import torch
from torch import nn

from cs336_basics.Decoder import BeamSearchDecoder


class DeterministicLanguageModel(nn.Module):
    """Toy language model whose globally best path differs from its greedy path."""

    vocab_size = 3

    def __init__(self):
        super().__init__()
        self.inputs = []
        self.grad_enabled_during_forward = []

    def forward(self, input_ids):
        self.inputs.append(input_ids.clone())
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())

        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(batch_size, sequence_length, self.vocab_size)

        if sequence_length == 2:
            # Token 1 is the greedy choice, while token 2 remains in the beam.
            logits[:, -1, :] = torch.tensor([0.0, 3.0, 2.0])
        else:
            # The path ending in token 2 gets a sufficiently strong continuation
            # to overtake the initially better path ending in token 1.
            for beam_index, last_token in enumerate(input_ids[:, -1].tolist()):
                if last_token == 2:
                    logits[beam_index, -1, :] = torch.tensor([5.0, 0.0, 0.0])

        return logits


def test_beam_search_uses_accumulated_scores():
    model = DeterministicLanguageModel()
    decoder = BeamSearchDecoder(model, beam_size=2, max_length=4)
    prompt = [2, 0]

    decoded_sequences = list(decoder.decode(prompt))

    assert len(decoded_sequences) == 2
    assert decoded_sequences[0].tolist() == [2, 0, 1]
    assert decoded_sequences[1].tolist() == [2, 0, 2, 0]

    # The second model call must contain both candidates retained by the beam.
    assert model.inputs[0].tolist() == [[2, 0]]
    assert {tuple(sequence) for sequence in model.inputs[1].tolist()} == {
        (2, 0, 1),
        (2, 0, 2),
    }

    # decode is decorated with torch.no_grad(), so inference must not build a graph.
    assert model.grad_enabled_during_forward == [False, False]
    assert all(not sequence.requires_grad for sequence in decoded_sequences)

    # Converting the prompt to a tensor internally must not mutate the caller's list.
    assert prompt == [2, 0]


def test_decode_stops_when_prompt_reaches_max_length():
    model = DeterministicLanguageModel()
    decoder = BeamSearchDecoder(model, beam_size=2, max_length=3)

    assert list(decoder.decode([0, 1, 2])) == []
    assert model.inputs == []


def test_decode_stops_when_best_sequence_reaches_eos_id():
    model = DeterministicLanguageModel()
    decoder = BeamSearchDecoder(model, eos_id=0, beam_size=2, max_length=6)

    decoded_sequences = list(decoder.decode([2, 0]))

    # The first generated token is returned normally. On the next step, the
    # globally best sequence ends in EOS, so decoding stops without yielding it.
    assert [sequence.tolist() for sequence in decoded_sequences] == [[2, 0, 1]]

    # max_length would allow four decoding steps, but EOS stops the model after two.
    assert len(model.inputs) == 2
    assert model.inputs[0].tolist() == [[2, 0]]
    assert {tuple(sequence) for sequence in model.inputs[1].tolist()} == {
        (2, 0, 1),
        (2, 0, 2),
    }