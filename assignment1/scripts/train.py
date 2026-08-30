import argparse
import torch
import numpy as np
from cs336_basics.Module import TransformerLM
from cs336_basics.Data import Loader

torch.manual_seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default='dataset')
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--context-length', default=256, type=int)
    args = parser.parse_args()

    
    dataset = np.memmap(
        f'{args.dataset_path}/TinyStoriesV2-GPT4-train.txt.npy',
        dtype=np.uint16,
        mode='r'
    )

    data_loader = Loader(
        dataset=dataset,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device='cpu'
    )

    model = TransformerLM(
        vocab_size=10000,
        context_length=args.context_length,
        num_layers=4,
        num_heads=16,
        d_model=512,
        d_ff=1344,
        rope_theta=10000
    )

    x, y = next(data_loader.get_batch_iterable())
    print(model(x))
