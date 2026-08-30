import argparse
import torch
import numpy as np
import os
from cs336_basics.Module import AdamW, CrossEntropyLoss
from cs336_basics.Module import TransformerLM
from cs336_basics.Data import Loader
from cs336_basics.Checkpoint import save_checkpoint

torch.manual_seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default='dataset')
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--context-length', default=256, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-every', default=1000, type=int)
    parser.add_argument('--log-dir', default='logs')
    args = parser.parse_args()


    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

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

        if step % args.save_every == 0:
            save_checkpoint(
                model, optimizer, step, f'{args.log_dir}/model{step}.pt'
            )

    save_checkpoint(
        model, optimizer, step, f'{args.log_dir}/model_final.pt'
    )