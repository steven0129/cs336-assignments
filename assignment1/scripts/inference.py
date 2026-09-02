import argparse
import pickle
from cs336_basics.Module import TransformerLM
from cs336_basics.Tokenizer import Tokenizer
from cs336_basics.Checkpoint import load_checkpoint
from cs336_basics.Decoder import BeamSearchDecoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-path', default='dataset')
    parser.add_argument('--prompt', default='Once upon a time there was a little boy named Ben.', type=str)
    parser.add_argument('--max-length', default=256, type=int)
    args = parser.parse_args()

    checkpoint = load_checkpoint('logs/model_final.pt')
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        num_heads=16,
        d_model=512,
        d_ff=1344,
        rope_theta=10000
    )

    model.load_state_dict(checkpoint['model'])

    with open(f"{args.vocab_path}/vocabs.pkl", "rb") as F:
        vocabs = pickle.load(F)

    with open(f"{args.vocab_path}/merges.pkl", "rb") as F:
        merges = pickle.load(F)

    bpe_tokenizer = Tokenizer(vocabs, merges, ["<|endoftext|>"])
    prompt_ids = bpe_tokenizer.encode(args.prompt)

    decoder = BeamSearchDecoder(
        model,
        beam_size=5,
        max_length=args.max_length,
        eos_id=bpe_tokenizer.token2tokenid[b"<|endoftext|>"]
    )

    for decoded_token_ids in decoder.decode(prompt_ids):
        decoded_text = bpe_tokenizer.decode(decoded_token_ids.tolist())
        print("\033[2J\033[H", end="")
        print(decoded_text)
