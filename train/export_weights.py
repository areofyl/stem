#!/usr/bin/env python3
# export model weights to a flat binary file for the C inference engine
# usage: python3 export_weights.py best.pt model.bin

import sys, struct
import torch
import numpy as np
from model_real import make_model, mel_band_edges

def main():
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} checkpoint.pt output.bin')
        sys.exit(1)

    ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    config = ckpt['config']

    model = make_model(config['size'])
    state = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model_state'].items()}
    model.load_state_dict(state)

    # collect all parameters in order
    params = []
    for name, param in model.named_parameters():
        params.append((name, param.detach().numpy()))

    # also need the band edges for reconstruction
    bands = model.band_split.bands

    with open(sys.argv[2], 'wb') as f:
        # header
        f.write(b'STEM')                                    # magic
        f.write(struct.pack('i', config['sample_rate']))    # sample rate
        f.write(struct.pack('i', 2048))                     # n_fft
        f.write(struct.pack('i', 512))                      # hop_length
        f.write(struct.pack('i', len(bands)))               # n_bands
        f.write(struct.pack('i', 128))                      # emb_dim (hardcoded for small)
        f.write(struct.pack('i', 4))                        # n_heads
        f.write(struct.pack('i', 4))                        # n_layers
        f.write(struct.pack('i', 4))                        # n_sources

        # band edges
        for start, end in bands:
            f.write(struct.pack('ii', start, end))

        # parameters: for each, write ndim, shape, then flat float32 data
        f.write(struct.pack('i', len(params)))
        for name, data in params:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('i', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('i', data.ndim))
            for s in data.shape:
                f.write(struct.pack('i', s))
            f.write(data.astype(np.float32).tobytes())

    size_mb = len(open(sys.argv[2], 'rb').read()) / 1e6
    print(f'exported {len(params)} tensors ({size_mb:.1f}MB)')
    print(f'bands: {len(bands)}')
    print(f'config: sr={config["sample_rate"]} sources={config["source_names"]}')

if __name__ == '__main__':
    main()
