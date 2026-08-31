#!/usr/bin/env python3
# download Slakh2100 and convert to the same format as get_musdb.py output
# 2100 synthesized songs with clean stems
#
# usage: python3 get_slakh.py /path/to/output

import sys, os, subprocess
from pathlib import Path
import soundfile as sf
import numpy as np

SAMPLE_RATE = 44100

# slakh has these instruments grouped into stems
# we map them to our 4 categories
SLAKH_TO_STEM = {
    'drums': 'drums',
    'bass': 'bass',
    'guitar': 'other',
    'piano': 'other',
    'strings': 'other',
    'synth_lead': 'vocals',   # closest thing to a lead voice in synth music
    'synth_pad': 'other',
    'reed': 'other',
    'brass': 'other',
    'flute': 'other',
    'organ': 'other',
    'chromatic_percussion': 'other',
}

def download_slakh(dest):
    """download slakh2100 using the official method"""
    # slakh is hosted on zenodo
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print('downloading Slakh2100 (this is ~90GB, will take a while)...')
    print('if this fails, download manually from https://zenodo.org/records/4599666')

    # try pip package first
    try:
        import slakh
        print('  using slakh python package')
        slakh.download(str(dest))
        return dest
    except ImportError:
        pass

    # fall back to wget from zenodo
    urls = [
        'https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz',
    ]
    for url in urls:
        fname = dest / url.split('/')[-1]
        if not fname.exists():
            print(f'  downloading {fname.name}...')
            subprocess.run(['wget', '-c', url, '-O', str(fname)], check=True)

        print(f'  extracting {fname.name}...')
        subprocess.run(['tar', 'xzf', str(fname), '-C', str(dest)], check=True)

    return dest


def convert_slakh(slakh_dir, output_dir):
    """convert slakh format to our stem format"""
    slakh_dir = Path(slakh_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # find the extracted directory
    candidates = list(slakh_dir.glob('slakh2100*')) + list(slakh_dir.glob('Track*'))
    if not candidates:
        # might be directly in slakh_dir
        candidates = [slakh_dir]

    # find all track directories
    tracks = []
    for base in candidates:
        for subset in ['train', 'test', 'validation', '']:
            search = base / subset if subset else base
            if search.exists():
                for d in sorted(search.iterdir()):
                    if d.is_dir() and (d / 'mix.flac').exists():
                        tracks.append(d)
                    elif d.is_dir() and (d / 'mix.wav').exists():
                        tracks.append(d)

    if not tracks:
        # try finding tracks recursively
        for mix_file in sorted(slakh_dir.rglob('mix.flac')):
            tracks.append(mix_file.parent)
        for mix_file in sorted(slakh_dir.rglob('mix.wav')):
            if mix_file.parent not in tracks:
                tracks.append(mix_file.parent)

    print(f'found {len(tracks)} tracks')

    converted = 0
    for i, track_dir in enumerate(tracks):
        name = track_dir.name
        out_dir = output_dir / f'slakh_{name}'

        if (out_dir / 'mix.wav').exists():
            continue

        # find mix
        mix_path = track_dir / 'mix.flac'
        if not mix_path.exists():
            mix_path = track_dir / 'mix.wav'
        if not mix_path.exists():
            continue

        try:
            mix_audio, sr = sf.read(str(mix_path), dtype='float32', always_2d=True)
        except Exception:
            continue

        # resample if needed
        if sr != SAMPLE_RATE:
            # use ffmpeg for resampling
            tmp = f'/tmp/slakh_resample_{name}.wav'
            subprocess.run([
                'ffmpeg', '-y', '-i', str(mix_path),
                '-ar', str(SAMPLE_RATE), '-ac', '2', tmp
            ], capture_output=True)
            mix_audio, _ = sf.read(tmp, dtype='float32', always_2d=True)
            os.remove(tmp)

        # collect stems
        stems = {'drums': None, 'bass': None, 'other': None, 'vocals': None}
        stems_dir = track_dir / 'stems'

        if stems_dir.exists():
            for stem_file in stems_dir.iterdir():
                if not stem_file.suffix in ['.flac', '.wav']:
                    continue
                stem_name = stem_file.stem.lower()

                # figure out which category this belongs to
                target = None
                for key, val in SLAKH_TO_STEM.items():
                    if key in stem_name:
                        target = val
                        break
                if target is None:
                    target = 'other'

                try:
                    audio, s_sr = sf.read(str(stem_file), dtype='float32', always_2d=True)
                    if s_sr != SAMPLE_RATE:
                        tmp = f'/tmp/slakh_stem_{name}_{stem_name}.wav'
                        subprocess.run([
                            'ffmpeg', '-y', '-i', str(stem_file),
                            '-ar', str(SAMPLE_RATE), '-ac', '2', tmp
                        ], capture_output=True)
                        audio, _ = sf.read(tmp, dtype='float32', always_2d=True)
                        os.remove(tmp)

                    # accumulate into the right category
                    if stems[target] is None:
                        stems[target] = audio
                    else:
                        min_len = min(stems[target].shape[0], audio.shape[0])
                        stems[target] = stems[target][:min_len] + audio[:min_len]
                except Exception:
                    continue

        # fill missing stems with silence
        n_samples = mix_audio.shape[0]
        for key in stems:
            if stems[key] is None:
                stems[key] = np.zeros((n_samples, 2), dtype='float32')
            elif stems[key].shape[0] != n_samples:
                if stems[key].shape[0] < n_samples:
                    stems[key] = np.pad(stems[key], ((0, n_samples - stems[key].shape[0]), (0, 0)))
                else:
                    stems[key] = stems[key][:n_samples]

        # save
        out_dir.mkdir(exist_ok=True)
        sf.write(str(out_dir / 'mix.wav'), mix_audio[:n_samples], SAMPLE_RATE)
        for key in stems:
            sf.write(str(out_dir / f'{key}.wav'), stems[key], SAMPLE_RATE)

        converted += 1
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(tracks)}] converted {converted} tracks')

    print(f'done: {converted} tracks converted to {output_dir}')


def main():
    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} /path/to/output')
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    # download to a temp location, convert to output
    raw_dir = output_dir / '_slakh_raw'
    download_slakh(raw_dir)
    convert_slakh(raw_dir, output_dir)

    print(f'\nslakh stems ready in {output_dir}')
    print('you can now retrain with the combined musdb + slakh data')

if __name__ == '__main__':
    main()
