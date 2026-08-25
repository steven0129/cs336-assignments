import os
import argparse
import pickle
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from cs336_basics.Module import TransformerLM
from cs336_basics.Tokenizer import BPETrainer, Tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict-path', default='dict')
    args = parser.parse_args()

    if not os.path.isdir(args.dict_path):
        os.makedirs(args.dict_path)

    if not os.path.isfile(f'{args.dict_path}/TinyStories-train.txt'):
        hf_hub_download(
            repo_id='roneneldan/TinyStories',
            repo_type="dataset",
            filename='TinyStories-train.txt',
            local_dir=args.dict_path
        )

    if not os.path.isfile(f"{args.dict_path}/vocabs.pkl") or \
        not os.path.isfile(f"{args.dict_path}/merges.pkl"):
        bpe_trainer = BPETrainer()
        print("Training BPE...")
        vocabs, merges = bpe_trainer.train(
            f'{args.dict_path}/TinyStories-train.txt',
            target_vocab_size=10000,
            special_tokens=["<|endoftext|>"]
        )

        with open(f"{args.dict_path}/vocabs.pkl", "wb") as F:
            pickle.dump(vocabs, F)

        with open(f"{args.dict_path}/merges.pkl", "wb") as F:
            pickle.dump(merges, F)

    with open(f"{args.dict_path}/vocabs.pkl", "rb") as F:
        vocabs = pickle.load(F)

    with open(f"{args.dict_path}/merges.pkl", "rb") as F:
        merges = pickle.load(F)

    print(f"Vocabulary Size: {len(vocabs)}")
    bpe_tokenizer = Tokenizer(vocabs, merges, ["<|endoftext|>"])
    token_ids = bpe_tokenizer.encode("Hello World!<|endoftext|>We are good!!")
    print(bpe_tokenizer.decode(token_ids))