from __future__ import annotations

from pathlib import Path
import json

import torch


def load_debug_epoch_batches(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        [[int(index) for index in batch] for batch in epoch]
        for epoch in payload["epoch_batches"]
    ]


class DebugEpochBatchLoader:
    def __init__(self, dataset, epoch_batches):
        self.dataset = dataset
        self.epoch_batches = [[int(index) for index in batch] for batch in epoch_batches]

    def __len__(self):
        return len(self.epoch_batches)

    def __iter__(self):
        for batch_indices in self.epoch_batches:
            xs = []
            ys = []
            for index in batch_indices:
                x, y = self.dataset[index]
                xs.append(x)
                ys.append(y)
            yield torch.stack(xs, dim=0), torch.stack(ys, dim=0)
