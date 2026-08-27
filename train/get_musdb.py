#!/usr/bin/env python3
# download MUSDB18-HQ and convert to the same format as batch_separate output
# so you can mix real stems with pseudo-labeled stems in training
#
# usage: python3 get_musdb.py /path/to/output

import sys, os
from pathlib import Path
import soundfile as sf
import numpy as np

def main():
    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} /path/to/output')
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import musdb
    except ImportError:
        print('installing musdb...')
        os.system('pip install musdb museval')
        import musdb

    # this downloads MUSDB18-HQ (~15GB) on first run
    print('loading MUSDB18 (will download on first run)...')
    db = musdb.DB(download=True, subsets='train')
    db_test = musdb.DB(download=True, subsets='test')

    all_tracks = list(db) + list(db_test)
    print(f'found {len(all_tracks)} tracks')

    source_map = {
        'drums': 'drums',
        'bass': 'bass',
        'other': 'other',
        'vocals': 'vocals',
    }

    for i, track in enumerate(all_tracks):
        name = track.name.replace('/', '_').replace('\\', '_')
        song_dir = output_dir / name

        if (song_dir / 'mix.wav').exists():
            print(f'  [{i+1}/{len(all_tracks)}] {name} (skipped)')
            continue

        print(f'  [{i+1}/{len(all_tracks)}] {name}')
        song_dir.mkdir(exist_ok=True)

        # save mix
        sr = track.rate
        sf.write(str(song_dir / 'mix.wav'), track.audio, sr)

        # save stems
        for stem_name, musdb_name in source_map.items():
            audio = track.targets[musdb_name].audio
            sf.write(str(song_dir / f'{stem_name}.wav'), audio, sr)

    print(f'\ndone: {len(all_tracks)} tracks in {output_dir}')
    print('these are real studio stems — perfect training labels')

if __name__ == '__main__':
    main()
