import numpy as np
import torch
import hydra
from pathlib import Path
from huggingface_hub import snapshot_download
from omegaconf import DictConfig
from tqdm import tqdm

from cs336_basics.Module import TransformerLM


def run_model(model):
    model(torch.randint(
        0, model.vocab_size,
        (1, model.context_length),
    ))


@hydra.main(version_base=None, config_path="../../assignment1/configs", config_name="config")
def main(config: DictConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run this benchmark on a machine with a CUDA-capable GPU.")

    model_config = config.model
    generation_config = config.generation
    benchmark_config = config.benchmark
    times = []

    model = TransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_model=model_config.d_model,
        d_ff=model_config.d_ff,
        rope_theta=model_config.rope_theta,
    )

    model = model.to(generation_config.device)
    model.eval()

    print('Warming up the model...')
    with torch.no_grad():
        for _ in tqdm(range(benchmark_config.num_warmup)):
            model(torch.randint(
                0, model_config.vocab_size,
                (1, model_config.context_length),
            ))

    torch.cuda.synchronize()

    print('Running benchmark...')
    with torch.no_grad():
        for _ in range(0, benchmark_config.num_trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()  # Start timing
            _ = model(torch.randint(
                0, model_config.vocab_size,
                (1, model_config.context_length),
            ))
            end_event.record()  # End timing

            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))  # Time in milliseconds

    
    mean_time = sum(times) / len(times)
    print(f"Average inference time over {benchmark_config.num_trials} steps: {mean_time:.4f} ms")



if __name__ == "__main__":
    main()
