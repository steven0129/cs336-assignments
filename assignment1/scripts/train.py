import argparse
import torch
import numpy as np
from cs336_basics.Module import AdamW, CrossEntropyLoss
from cs336_basics.Module import TransformerLM
from cs336_basics.Data import Loader

torch.manual_seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default='dataset')
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--context-length', default=256, type=int)
    parser.add_argument('--device', default='cuda')
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
        device=args.device
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

    model = model.to(args.device)
    model.train()
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    for step, (x, y) in enumerate(data_loader.get_batch_iterable()):
        optimizer.zero_grad()

        logits = model(x)  # (batch_size, context_length, vocab_size)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(
                f"step={step:6d}/{len(data_loader)}, loss={loss.item():.4f}"
            )