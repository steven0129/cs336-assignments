import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from cs336_basics.Module import TransformerLM
from cs336_basics.Tokenizer import BPETrainer, Tokenizer
from cs336_basics.Data import Loader


RAW_TINYSTORIES_TXT = f"TinyStoriesV2-GPT4-train.txt"
CONTEXT_LENGTH = 256
BATCH_SIZE = 5


if __name__ == '__main__':
    data_loader = Loader()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict-path', default='dict')
    args = parser.parse_args()

    if not os.path.isdir(args.dict_path):
        os.makedirs(args.dict_path)

    if not os.path.isfile(f'{args.dict_path}/{RAW_TINYSTORIES_TXT}'):
        hf_hub_download(
            repo_id='roneneldan/TinyStories',
            repo_type="dataset",
            filename=RAW_TINYSTORIES_TXT,
            local_dir=args.dict_path
        )

    if not os.path.isfile(f"{args.dict_path}/vocabs.pkl") or \
        not os.path.isfile(f"{args.dict_path}/merges.pkl"):
        bpe_trainer = BPETrainer()
        print("Training BPE...")
        vocabs, merges = bpe_trainer.train(
            f'{args.dict_path}/{RAW_TINYSTORIES_TXT}',
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

    if not os.path.isfile(f"{args.dict_path}/{RAW_TINYSTORIES_TXT}.bin"):
        print("Building numpy array of TinyStories...")
        with open(f'{args.dict_path}/{RAW_TINYSTORIES_TXT}') as F:
            counter = 0
            queue = []
            token_ids = bpe_tokenizer.encode_iterable(F, special_tokens=["<|endoftext|>"])
            for token_id in tqdm(token_ids):
                counter += 1
                queue.append(token_id)
                if counter == 100 * 1024 * 1024:
                    queue = np.array(queue, dtype=np.uint16)
                    with open(f"{args.dict_path}/{RAW_TINYSTORIES_TXT}.bin", "ab") as F:
                        queue.tofile(F)

                    queue = []
