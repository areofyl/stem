#!/usr/bin/env python3
# dataset loader
# expects: data_dir/song_name/{mix,vocals,drums,bass,other}.wav

import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np

SOURCES = ['drums', 'bass', 'other', 'vocals']


def train_val_split(data_dir, val_fraction=0.1):
    """split songs into train/val lists (reproducible)"""
    data_dir = Path(data_dir)
    all_songs = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / 'mix.wav').exists() and all((d / f'{s}.wav').exists() for s in SOURCES):
            all_songs.append(d.name)

    random.seed(42)
    random.shuffle(all_songs)
    n_val = max(1, int(len(all_songs) * val_fraction))
    return all_songs[n_val:], all_songs[:n_val]


class StemDataset(Dataset):
    def __init__(self, data_dir, chunk_length=10, sample_rate=44100, augment=True, songs=None):
        self.data_dir = Path(data_dir)
        self.chunk_samples = chunk_length * sample_rate
        self.sr = sample_rate
        self.augment = augment

        if songs is not None:
            self.songs = [self.data_dir / s for s in songs]
        else:
            self.songs = []
            for d in sorted(self.data_dir.iterdir()):
                if not d.is_dir():
                    continue
                if (d / 'mix.wav').exists() and all((d / f'{s}.wav').exists() for s in SOURCES):
                    self.songs.append(d)

        print(f'  dataset: {len(self.songs)} songs')

    def __len__(self):
        return len(self.songs) * 10

    def load_stems(self, song_dir):
        stems = {}
        for s in SOURCES:
            audio, _ = sf.read(str(song_dir / f'{s}.wav'), dtype='float32', always_2d=True)
            stems[s] = audio.T  # (2, samples)
        return stems

    def __getitem__(self, idx):
        song_dir = self.songs[idx % len(self.songs)]
        stems = self.load_stems(song_dir)
        total_samples = stems[SOURCES[0]].shape[1]

        # random chunk
        if total_samples > self.chunk_samples:
            start = random.randint(0, total_samples - self.chunk_samples)
            end = start + self.chunk_samples
        else:
            start, end = 0, total_samples

        stem_arrays = [stems[s][:, start:end] for s in SOURCES]

        if self.augment:
            # full remix: sample every stem from a different song (50% chance)
            # with N songs this gives N^4 combinations from the same data
            if random.random() < 0.5:
                for i, src in enumerate(SOURCES):
                    donor = random.choice(self.songs)
                    donor_stems = self.load_stems(donor)
                    donor_len = donor_stems[src].shape[1]
                    if donor_len >= self.chunk_samples:
                        ds = random.randint(0, donor_len - self.chunk_samples)
                        stem_arrays[i] = donor_stems[src][:, ds:ds+self.chunk_samples]

            # random gain per source (+/- 6dB)
            for i in range(len(SOURCES)):
                gain = 10 ** (random.uniform(-6, 6) / 20)
                stem_arrays[i] = stem_arrays[i] * gain

            # channel swap
            if random.random() < 0.5:
                stem_arrays = [s[::-1].copy() for s in stem_arrays]

        # build mix from (possibly augmented) stems
        stem_stack = np.stack(stem_arrays)  # (n_sources, 2, samples)
        mix = stem_stack.sum(axis=0)  # (2, samples)

        # pad short songs
        if mix.shape[1] < self.chunk_samples:
            pad = self.chunk_samples - mix.shape[1]
            mix = np.pad(mix, ((0, 0), (0, pad)))
            stem_stack = np.pad(stem_stack, ((0, 0), (0, 0), (0, pad)))

        return torch.from_numpy(mix.copy()), torch.from_numpy(stem_stack.copy())
