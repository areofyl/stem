#!/usr/bin/env python3
# dataset loader for training
# expects the output of batch_separate.py:
#   data_dir/song_name/mix.wav
#   data_dir/song_name/vocals.wav
#   data_dir/song_name/drums.wav
#   data_dir/song_name/bass.wav
#   data_dir/song_name/other.wav

import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np

SOURCES = ['drums', 'bass', 'other', 'vocals']

class StemDataset(Dataset):
    def __init__(self, data_dir, chunk_length=10, sample_rate=44100, augment=True):
        self.data_dir = Path(data_dir)
        self.chunk_samples = chunk_length * sample_rate
        self.sr = sample_rate
        self.augment = augment

        # find all valid song directories
        self.songs = []
        for d in sorted(self.data_dir.iterdir()):
            if not d.is_dir():
                continue
            mix = d / 'mix.wav'
            stems = [d / f'{s}.wav' for s in SOURCES]
            if mix.exists() and all(s.exists() for s in stems):
                self.songs.append(d)

        print(f'  dataset: {len(self.songs)} songs from {data_dir}')

    def __len__(self):
        # each epoch, sample a bunch of random chunks
        return len(self.songs) * 10

    def __getitem__(self, idx):
        song_dir = self.songs[idx % len(self.songs)]

        # load mix
        mix, _ = sf.read(str(song_dir / 'mix.wav'), dtype='float32', always_2d=True)
        total_samples = mix.shape[0]

        # random chunk
        if total_samples > self.chunk_samples:
            start = random.randint(0, total_samples - self.chunk_samples)
            end = start + self.chunk_samples
        else:
            start = 0
            end = total_samples

        mix_chunk = mix[start:end].T  # (channels, samples)

        # load corresponding stem chunks
        stems = []
        for source in SOURCES:
            stem, _ = sf.read(str(song_dir / f'{source}.wav'), dtype='float32', always_2d=True)
            stems.append(stem[start:end].T)

        stems = np.stack(stems)  # (n_sources, channels, samples)

        # augmentation
        if self.augment:
            # random gain per source (+/- 3dB)
            for i in range(len(SOURCES)):
                gain = 10 ** (random.uniform(-3, 3) / 20)
                stems[i] *= gain

            # remix: sum stems to create augmented mix
            mix_chunk = stems.sum(axis=0)

            # random channel swap
            if random.random() < 0.5:
                mix_chunk = mix_chunk[::-1].copy()
                stems = stems[:, ::-1].copy()

        # pad if needed
        if mix_chunk.shape[1] < self.chunk_samples:
            pad = self.chunk_samples - mix_chunk.shape[1]
            mix_chunk = np.pad(mix_chunk, ((0, 0), (0, pad)))
            stems = np.pad(stems, ((0, 0), (0, 0), (0, pad)))

        return torch.from_numpy(mix_chunk), torch.from_numpy(stems)
