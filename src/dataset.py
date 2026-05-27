import random
import torch
from torch.utils.data import Dataset

import random
import torch
from torch.utils.data import Dataset


class TxRandomWindowDataset(Dataset):
    def __init__(self, out_seqs, tree_seqs, ctx_seqs, block_size: int, num_samples_per_epoch: int, pad_id: int = 0):
        self.out_seqs = out_seqs
        self.tree_seqs = tree_seqs
        self.ctx_seqs = ctx_seqs
        self.T = block_size
        self.num_samples_per_epoch = num_samples_per_epoch
        self.pad_id = pad_id

        self.valid_j = list(range(len(out_seqs)))

    def __len__(self):
        return self.num_samples_per_epoch

    def __getitem__(self, _):
        j = random.choice(self.valid_j)

        out = self.out_seqs[j]
        tree = self.tree_seqs[j]
        ctx = self.ctx_seqs[j]

        required_len = self.T + 1
        current_len = len(out)

        if current_len < required_len:
            # padding
            pad_size = required_len - current_len
            out = torch.cat([out, torch.full((pad_size,), self.pad_id, dtype=torch.long)])
            tree = torch.cat([tree, torch.full((pad_size,), self.pad_id, dtype=torch.long)])
            ctx = torch.cat([ctx, torch.full((pad_size,), self.pad_id, dtype=torch.long)])
            current_len = required_len

        start = random.randint(0, current_len - self.T - 1)

        x_out = out[start:start+self.T]
        x_tree = tree[start:start+self.T]
        x_ctx = ctx[start:start+self.T]
        y = out[start+1:start+1+self.T]

        return x_out, x_tree, x_ctx, y

class TxStrideWindowDataset(Dataset):
    def __init__(self, out_seqs, tree_seqs, ctx_seqs, block_size: int, stride: int = None):
        assert len(out_seqs) == len(tree_seqs) == len(ctx_seqs)
        self.out_seqs = out_seqs
        self.tree_seqs = tree_seqs
        self.ctx_seqs = ctx_seqs
        self.T = block_size
        self.stride = stride if stride is not None else block_size

        self.samples = []
        for j, out in enumerate(out_seqs):
            if len(out) <= self.T + 1:
                continue
            for start in range(0, len(out) - self.T - 1, self.stride):
                self.samples.append((j, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        j, start = self.samples[i]
        out = self.out_seqs[j]
        tree = self.tree_seqs[j]
        ctx = self.ctx_seqs[j]

        x_out  = out[start:start+self.T]
        x_tree = tree[start:start+self.T]
        x_ctx  = ctx[start:start+self.T]
        y      = out[start+1:start+1+self.T]
        return x_out, x_tree, x_ctx, y

