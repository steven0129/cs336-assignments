import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from cs336_basics.Tokenizer import BPETrainer, Tokenizer


RAW_TINYSTORIES_TXT = f"TinyStoriesV2-GPT4-train.txt"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default='dataset')
    parser.add_argument('--context-length', default=256, type=int)
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        os.makedirs(args.dataset_path)

    if not os.path.isfile(f'{args.dataset_path}/{RAW_TINYSTORIES_TXT}'):
        hf_hub_download(
            repo_id='roneneldan/TinyStories',
            repo_type="dataset",
            filename=RAW_TINYSTORIES_TXT,
            local_dir=args.dataset_path
        )

    if not os.path.isfile(f"{args.dataset_path}/vocabs.pkl") or \
        not os.path.isfile(f"{args.dataset_path}/merges.pkl"):
        bpe_trainer = BPETrainer()
        print("Training BPE...")
        vocabs, merges = bpe_trainer.train(
            f'{args.dataset_path}/{RAW_TINYSTORIES_TXT}',
            target_vocab_size=10000,
            special_tokens=["<|endoftext|>"]
        )

        with open(f"{args.dataset_path}/vocabs.pkl", "wb") as F:
            pickle.dump(vocabs, F)

        with open(f"{args.dataset_path}/merges.pkl", "wb") as F:
            pickle.dump(merges, F)

    with open(f"{args.dataset_path}/vocabs.pkl", "rb") as F:
        vocabs = pickle.load(F)

    with open(f"{args.dataset_path}/merges.pkl", "rb") as F:
        merges = pickle.load(F)

    print(f"Vocabulary Size: {len(vocabs)}")
    bpe_tokenizer = Tokenizer(vocabs, merges, ["<|endoftext|>"])

    if not os.path.isfile(f"{args.dataset_path}/{RAW_TINYSTORIES_TXT}.npy"):
        print(f"Building {args.dataset_path}/{RAW_TINYSTORIES_TXT}.npy...")
        with open(f'{args.dataset_path}/{RAW_TINYSTORIES_TXT}') as F:
            counter = 0
            queue = []
            token_ids = bpe_tokenizer.encode_iterable(F, special_tokens=["<|endoftext|>"])
            for token_id in tqdm(token_ids):
                counter += 1
                queue.append(token_id)
                if counter == 100 * 1024 * 1024:
                    queue = np.array(queue, dtype=np.uint16)
                    with open(f"{args.dataset_path}/{RAW_TINYSTORIES_TXT}.npy", "ab") as F:
                        queue.tofile(F)

                    queue = []
                    counter = 0

            if queue:
                with open(f"{args.dataset_path}/{RAW_TINYSTORIES_TXT}.npy", "ab") as F:
                    queue = np.array(queue, dtype=np.uint16)
                    queue.tofile(F)
