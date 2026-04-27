import torch

import torch.nn as nn


def to_freqs(x, magn, num_freqs, dim):
    r = x.clone()
    f = []
    for _ in range(num_freqs):
        f.append(r / magn)
        r -= r // magn
        magn = magn / 2
    return torch.cat(f, dim)


def log_freq(x, N_freqs, dim):
    include_input = True

    max_freq = N_freqs - 1
    # num_freqs= multires
    log_sampling = True
    periodic_fns = [torch.sin, torch.cos]

    embed_fns = []
    d = x.shape[dim]
    out_dim = 0
    if include_input:
        embed_fns.append(lambda x: x)
        out_dim += d
    # freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
    if log_sampling:
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
    else:
        freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)
    for freq in freq_bands:
        for p_fn in periodic_fns:
            embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
            out_dim += d

    embed_fns = embed_fns
    out_dim = out_dim

    return torch.cat([fn(x) for fn in embed_fns], -1)
