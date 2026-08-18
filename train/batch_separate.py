#!/usr/bin/env python3
# batch separate a directory of songs into stems using the best available models
# run this on the cloud box first, then train on the output
#
# usage: python3 batch_separate.py /path/to/music /path/to/output
#
# output structure:
#   output/
#     song1/
#       mix.wav
#       vocals.wav
#       drums.wav
#       bass.wav
#       other.wav
#     song2/
#       ...

import sys, os, shutil, subprocess
from pathlib import Path
import torch
import soundfile as sf

SAMPLE_RATE = 44100
EXTENSIONS = {'.mp3', '.flac', '.wav', '.opus', '.ogg', '.m4a', '.aac', '.wma'}

def find_songs(music_dir):
    songs = []
    for root, _, files in os.walk(music_dir):
        for f in files:
            if Path(f).suffix.lower() in EXTENSIONS:
                songs.append(Path(root) / f)
    songs.sort()
    return songs

def convert_to_wav(input_path, output_path):
    """convert any audio format to 44.1khz stereo wav"""
    subprocess.run([
        'ffmpeg', '-y', '-i', str(input_path),
        '-ar', str(SAMPLE_RATE), '-ac', '2',
        str(output_path)
    ], capture_output=True)

def separate_demucs(mix_path, output_dir, device):
    """separate with htdemucs_6s for instruments"""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import save_audio

    model = get_model('htdemucs_6s')
    if device == 'cuda':
        model.cuda()
    model.eval()

    audio, _ = sf.read(str(mix_path), dtype='float32', always_2d=True)
    audio = torch.from_numpy(audio.T)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)

    ref = audio.mean(0)
    mean, std = ref.mean(), ref.std()
    audio = (audio - mean) / std

    if device == 'cuda':
        audio = audio.cuda()

    with torch.no_grad():
        sources = apply_model(model, audio[None], progress=False)[0]

    if device == 'cuda':
        sources = sources.cpu()
    sources = sources * std + mean

    for i, name in enumerate(model.sources):
        save_audio(sources[i], str(output_dir / f'{name}.wav'), SAMPLE_RATE)

    return model.sources

def main():
    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} /path/to/music /path/to/output')
        sys.exit(1)

    music_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    songs = find_songs(music_dir)
    print(f'found {len(songs)} songs')

    done = 0
    failed = 0

    for i, song in enumerate(songs):
        name = song.stem
        song_dir = output_dir / name

        # skip already processed
        if (song_dir / 'mix.wav').exists():
            print(f'  [{i+1}/{len(songs)}] {name} (skipped, already done)')
            done += 1
            continue

        print(f'  [{i+1}/{len(songs)}] {name}')
        song_dir.mkdir(exist_ok=True)

        try:
            # convert to wav
            mix_path = song_dir / 'mix.wav'
            convert_to_wav(song, mix_path)

            # separate
            separate_demucs(mix_path, song_dir, device)
            done += 1
        except Exception as e:
            print(f'    failed: {e}')
            shutil.rmtree(song_dir, ignore_errors=True)
            failed += 1

    print(f'\ndone: {done} songs, {failed} failed')
    print(f'output: {output_dir}')

if __name__ == '__main__':
    main()
