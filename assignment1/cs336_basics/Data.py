import torch
import numpy as np


class Loader():
    def get_batch(self, dataset, batch_size, context_length, device):
        starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
        x = torch.stack([
            torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
            for i in starting_idxs
        ])

        y = torch.stack(
            [
                torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
                for i in starting_idxs
            ]
        )  # fmt: skip
        if "cuda" in device:
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y