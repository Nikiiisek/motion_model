# collate_varlen.py
import torch

def pad_collate_varlen(batch):
    # batch: list of (x: TxCxHxW, y, length)
    xs, ys, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)

    B = len(xs)
    Tmax = int(lengths.max().item())
    C, H, W = xs[0].shape[1], xs[0].shape[2], xs[0].shape[3]

    x_pad = torch.zeros(B, Tmax, C, H, W, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        t = x.shape[0]
        x_pad[i, :t] = x

    y = torch.stack(ys, dim=0)
    return x_pad, y, lengths