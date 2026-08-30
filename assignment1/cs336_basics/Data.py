import torch
import numpy as np


class Loader():
    def __init__(self, dataset, batch_size, context_length, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def get_batch(self):
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
        return x, y