#!/usr/bin/env python3
# run the trained stem model on a song
# usage: python3 inference.py best.pt input.wav output_dir

import sys, os
import torch
import soundfile as sf
from model import make_model

def main():
    if len(sys.argv) < 4:
        print(f'usage: {sys.argv[0]} model.pt input.wav output_dir')
        sys.exit(1)

    model_path = sys.argv[1]
    input_path = sys.argv[2]
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)

    # load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    model = make_model(config['size'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # load audio
    audio, sr = sf.read(input_path, dtype='float32', always_2d=True)
    audio = torch.from_numpy(audio.T)  # (channels, samples)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)

    # separate
    print('  separating...')
    sources = model.separate(audio)

    # save
    for i, name in enumerate(config['source_names']):
        out_path = os.path.join(output_dir, f'{name}.wav')
        sf.write(out_path, sources[i].numpy().T, config['sample_rate'])
        print(f'  saved {name}')

    print(f'  done: {len(config["source_names"])} stems')

if __name__ == '__main__':
    main()
