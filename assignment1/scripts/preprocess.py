import pickle
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from cs336_basics.config_utils import load_config
from cs336_basics.Tokenizer import BPETrainer, Tokenizer


def main():
    config, _ = load_config("Download and tokenize a text dataset.")
    data_config = config.data

    dataset_path = Path(data_config.dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)

    raw_dataset_path = dataset_path / data_config.raw_filename
    valid_dataset_path = dataset_path / data_config.valid_filename
    vocab_path = dataset_path / data_config.vocab_filename
    merges_path = dataset_path / data_config.merges_filename
    tokenized_path = dataset_path / data_config.tokenized_filename
    valid_tokenized_path = dataset_path / data_config.valid_tokenized_filename

    if not raw_dataset_path.is_file():
        hf_hub_download(
            repo_id=data_config.huggingface.repo_id,
            repo_type=data_config.huggingface.repo_type,
            filename=data_config.raw_filename,
            local_dir=dataset_path,
        )

    if not vocab_path.is_file() or not merges_path.is_file():
        bpe_trainer = BPETrainer()
        print("Training BPE...")
        vocabs, merges = bpe_trainer.train(
            raw_dataset_path,
            target_vocab_size=data_config.target_vocab_size,
            special_tokens=list(data_config.special_tokens),
        )

        with vocab_path.open("wb") as file:
            pickle.dump(vocabs, file)

        with merges_path.open("wb") as file:
            pickle.dump(merges, file)

    with vocab_path.open("rb") as file:
        vocabs = pickle.load(file)

    with merges_path.open("rb") as file:
        merges = pickle.load(file)

    print(f"Vocabulary Size: {len(vocabs)}")
    special_tokens = list(data_config.special_tokens)
    bpe_tokenizer = Tokenizer(vocabs, merges, special_tokens)

    if not tokenized_path.is_file():
        print(f"Building {tokenized_path}...")
        with raw_dataset_path.open() as file:
            counter = 0
            queue = []
            token_ids = bpe_tokenizer.encode_iterable(file, special_tokens=special_tokens)
            for token_id in tqdm(token_ids):
                counter += 1
                queue.append(token_id)
                if counter == data_config.write_buffer_size:
                    token_buffer = np.array(queue, dtype=data_config.dtype)
                    with tokenized_path.open("ab") as output_file:
                        token_buffer.tofile(output_file)

                    queue = []
                    counter = 0

            if queue:
                token_buffer = np.array(queue, dtype=data_config.dtype)
                with tokenized_path.open("ab") as output_file:
                    token_buffer.tofile(output_file)

    if not valid_tokenized_path.is_file():
        print(f"Building {valid_tokenized_path}...")
        with valid_dataset_path.open() as file:
            counter = 0
            queue = []
            token_ids = bpe_tokenizer.encode_iterable(file, special_tokens=special_tokens)
            for token_id in tqdm(token_ids):
                counter += 1
                queue.append(token_id)
                if counter == data_config.write_buffer_size:
                    token_buffer = np.array(queue, dtype=data_config.dtype)
                    with valid_tokenized_path.open("ab") as output_file:
                        token_buffer.tofile(output_file)

                    queue = []
                    counter = 0

            if queue:
                token_buffer = np.array(queue, dtype=data_config.dtype)
                with valid_tokenized_path.open("ab") as output_file:
                    token_buffer.tofile(output_file)


if __name__ == "__main__":
    main()
