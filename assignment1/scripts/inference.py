import pickle
from pathlib import Path

from huggingface_hub import snapshot_download

from cs336_basics.Checkpoint import load_checkpoint
from cs336_basics.config_utils import load_config
from cs336_basics.Decoder import BeamSearchDecoder
from cs336_basics.Module import TransformerLM
from cs336_basics.Tokenizer import Tokenizer


def main():
    config, _ = load_config("Generate text with a trained Transformer language model.")
    data_config = config.data
    model_config = config.model
    generation_config = config.generation

    snapshot_download(
        repo_id=generation_config.huggingface.dataset_repo_id,
        repo_type="dataset",
        local_dir=generation_config.huggingface.dataset_local_dir,
    )

    snapshot_download(
        repo_id=generation_config.huggingface.model_repo_id,
        repo_type="model",
        local_dir=generation_config.huggingface.model_local_dir,
    )

    checkpoint = load_checkpoint(generation_config.checkpoint_path)
    model = TransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_model=model_config.d_model,
        d_ff=model_config.d_ff,
        rope_theta=model_config.rope_theta,
    )

    model.load_state_dict(checkpoint["model"])
    model = model.to(generation_config.device)
    model.eval()

    dataset_path = Path(data_config.dataset_path)
    with (dataset_path / data_config.vocab_filename).open("rb") as file:
        vocabs = pickle.load(file)

    with (dataset_path / data_config.merges_filename).open("rb") as file:
        merges = pickle.load(file)

    bpe_tokenizer = Tokenizer(vocabs, merges, list(data_config.special_tokens))
    prompt_ids = bpe_tokenizer.encode(generation_config.prompt)

    decoder = BeamSearchDecoder(
        model,
        beam_size=generation_config.beam_size,
        max_length=generation_config.max_length,
        eos_id=bpe_tokenizer.token2tokenid[data_config.special_tokens[0].encode()],
    )

    for decoded_token_ids in decoder.decode(prompt_ids):
        decoded_text = bpe_tokenizer.decode(decoded_token_ids.tolist())
        print("\033[2J\033[H", end="")
        print(decoded_text)


if __name__ == "__main__":
    main()
