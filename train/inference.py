#!/usr/bin/env python3
# run the trained stem model on a song
# usage: python3 inference.py best.pt input.wav output_dir

import sys, os, time
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

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    model = make_model(config['size'])

    # torch.compile prefixes keys with _orig_mod., strip it for loading
    state = checkpoint['model_state']
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # convert to 44100Hz wav first if needed
    import subprocess, tempfile
    tmp_wav = os.path.join(tempfile.gettempdir(), 'stem_inference_input.wav')
    subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-ar', str(config['sample_rate']), '-ac', '2', tmp_wav
    ], capture_output=True)

    audio, sr = sf.read(tmp_wav, dtype='float32', always_2d=True)
    os.remove(tmp_wav)
    audio = torch.from_numpy(audio.T)

    duration = audio.shape[1] / config['sample_rate']
    print(f'separating {duration:.1f}s of audio...')

    t0 = time.time()
    sources = model.separate(audio)
    elapsed = time.time() - t0

    print(f'done in {elapsed:.1f}s ({duration/elapsed:.1f}x realtime)')

    for i, name in enumerate(config['source_names']):
        out_path = os.path.join(output_dir, f'{name}.wav')
        sf.write(out_path, sources[i].numpy().T, config['sample_rate'])
        print(f'  saved {name}')

if __name__ == '__main__':
    main()
