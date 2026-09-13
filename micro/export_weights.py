#!/usr/bin/env python3
# export micro model weights to binary for C inference
# usage: python3 export_weights.py checkpoints/vocals.pt vocals.bin

import sys, struct
import torch
import numpy as np

def main():
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} checkpoint.pt output.bin')
        sys.exit(1)

    ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    state = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model_state'].items()}

    with open(sys.argv[2], 'wb') as f:
        # header
        f.write(b'MICR')
        f.write(struct.pack('i', 512))   # n_fft
        f.write(struct.pack('i', 256))   # hop
        f.write(struct.pack('i', 128))   # hidden
        f.write(struct.pack('i', 2))     # n_gru_layers

        # write tensors in order the C code expects
        names = [
            'encoder.0.weight', 'encoder.0.bias',
            'encoder.2.weight', 'encoder.2.bias',
            'gru.weight_ih_l0', 'gru.weight_hh_l0',
            'gru.bias_ih_l0', 'gru.bias_hh_l0',
            'gru.weight_ih_l1', 'gru.weight_hh_l1',
            'gru.bias_ih_l1', 'gru.bias_hh_l1',
            'decoder.0.weight', 'decoder.0.bias',
            'decoder.2.weight', 'decoder.2.bias',
        ]

        f.write(struct.pack('i', len(names)))
        for name in names:
            data = state[name].numpy().astype(np.float32)
            name_bytes = name.encode()
            f.write(struct.pack('i', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('i', data.ndim))
            for s in data.shape:
                f.write(struct.pack('i', s))
            f.write(data.tobytes())

    import os
    size = os.path.getsize(sys.argv[2]) / 1e3
    print(f'exported {len(names)} tensors ({size:.0f}KB) to {sys.argv[2]}')

if __name__ == '__main__':
    main()
