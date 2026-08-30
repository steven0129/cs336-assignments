import torch
import numpy as np


class Loader():
    def __init__(self, dataset, batch_size, context_length, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.length = len(dataset) // (batch_size * context_length) + 1

    def get_batch(self):
        return next(self.get_batch_iterable())

    def get_batch_iterable(self):
        for _ in range(self.length):
            starting_idxs = torch.randint(len(self.dataset) - self.context_length, (self.batch_size,))
            x = torch.stack([
                torch.from_numpy((self.dataset[i : i + self.context_length]).astype(np.int64))
                for i in starting_idxs
            ])

            y = torch.stack(
                [
                    torch.from_numpy((self.dataset[i + 1 : i + 1 + self.context_length]).astype(np.int64))
                    for i in starting_idxs
                ]
            )  # fmt: skip
            if "cuda" in self.device:
                x = x.pin_memory().to(self.device, non_blocking=True)
                y = y.pin_memory().to(self.device, non_blocking=True)
            else:
                x = x.to(self.device)
                y = y.to(self.device)
            yield x, y

    def __len__(self):
        return self.length