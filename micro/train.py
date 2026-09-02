#!/usr/bin/env python3
# train a micro separator for one instrument
#
# python3 train.py --data /path/to/musdb --target vocals --epochs 100
# python3 train.py --data /path/to/musdb --target drums --epochs 100

import argparse, os, time, math, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
from pathlib import Path
from model import make_model

SOURCES = ['drums', 'bass', 'other', 'vocals']

class StemDataset(Dataset):
    def __init__(self, data_dir, target, chunk_samples=44100*3, augment=True, songs=None):
        self.data_dir = Path(data_dir)
        self.target = target
        self.chunk = chunk_samples
        self.augment = augment

        if songs:
            self.songs = [self.data_dir / s for s in songs]
        else:
            self.songs = []
            for d in sorted(self.data_dir.iterdir()):
                if d.is_dir() and (d / 'mix.wav').exists() and (d / f'{target}.wav').exists():
                    self.songs.append(d)

        print(f'  dataset: {len(self.songs)} songs, target: {target}')

    def __len__(self):
        return len(self.songs) * 20

    def load(self, path):
        a, _ = sf.read(str(path), dtype='float32', always_2d=True)
        return a.mean(axis=1)  # mono

    def __getitem__(self, idx):
        song = self.songs[idx % len(self.songs)]

        # load mix and target
        mix = self.load(song / 'mix.wav')
        target = self.load(song / f'{self.target}.wav')

        # random chunk
        if len(mix) > self.chunk:
            start = random.randint(0, len(mix) - self.chunk)
            mix = mix[start:start+self.chunk]
            target = target[start:start+self.chunk]
        else:
            mix = np.pad(mix, (0, self.chunk - len(mix)))
            target = np.pad(target, (0, self.chunk - len(target)))

        if self.augment:
            # random gain
            gain = 10 ** (random.uniform(-6, 6) / 20)
            mix = mix * gain
            target = target * gain

            # remix: rebuild mix from random stems (50% chance)
            if random.random() < 0.5:
                donor = random.choice(self.songs)
                new_mix = np.zeros_like(mix)
                new_target = None
                for src in SOURCES:
                    s = self.load(donor / f'{src}.wav')
                    if len(s) > self.chunk:
                        st = random.randint(0, len(s) - self.chunk)
                        s = s[st:st+self.chunk]
                    else:
                        s = np.pad(s, (0, self.chunk - len(s)))
                    g = 10 ** (random.uniform(-6, 6) / 20)
                    s = s * g
                    new_mix += s
                    if src == self.target:
                        new_target = s
                mix = new_mix
                target = new_target

        return torch.from_numpy(mix).float(), torch.from_numpy(target).float()


def train_val_split(data_dir, target, val_frac=0.1):
    data_dir = Path(data_dir)
    songs = [d.name for d in sorted(data_dir.iterdir())
             if d.is_dir() and (d / 'mix.wav').exists() and (d / f'{target}.wav').exists()]
    random.seed(42)
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * val_frac))
    return songs[n_val:], songs[:n_val]


def stft_loss(pred, target, fft_sizes=[512, 1024, 2048]):
    loss = 0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)
        p = torch.stft(pred, n_fft, hop, window=window, return_complex=True).abs()
        t = torch.stft(target, n_fft, hop, window=window, return_complex=True).abs()
        loss += (p - t).abs().mean()
        loss += (p.clamp(min=1e-7).log() - t.clamp(min=1e-7).log()).abs().mean()
    return loss / len(fft_sizes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--target', required=True, choices=SOURCES)
    parser.add_argument('--output', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    device = 'cpu'
    print(f'training micro separator for: {args.target}')

    model = make_model().to(device)
    torch.set_num_threads(8)

    train_songs, val_songs = train_val_split(args.data, args.target)
    train_set = StemDataset(args.data, args.target, songs=train_songs)
    val_set = StemDataset(args.data, args.target, songs=val_songs, augment=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    warmup = len(train_loader) * 5
    total = len(train_loader) * args.epochs
    def lr_fn(step):
        if step < warmup: return step / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    os.makedirs(args.output, exist_ok=True)
    best_val = float('inf')

    total_batches = len(train_loader)
    print(f'epochs: {args.epochs}, batches/epoch: {total_batches}')
    print()

    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0
        n = 0
        t0 = time.time()

        for mix, target in train_loader:
            pred = model(mix)
            loss = stft_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            n += 1
            lr = optimizer.param_groups[0]['lr']
            print(f'\r  epoch {epoch+1}/{args.epochs}  batch {n}/{total_batches}  loss: {ep_loss/n:.4f}  lr: {lr:.6f}', end='', flush=True)

        # validate
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for mix, target in val_loader:
                pred = model(mix)
                val_loss += stft_loss(pred, target).item()
                vn += 1
        val_loss /= max(vn, 1)
        elapsed = time.time() - t0

        print(f'\r  epoch {epoch+1}/{args.epochs}  train: {ep_loss/n:.4f}  val: {val_loss:.4f}  {elapsed:.0f}s')

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'target': args.target,
                'val_loss': val_loss,
            }, os.path.join(args.output, f'{args.target}.pt'))
            print(f'    ^ new best')

    print(f'\ndone. best val loss: {best_val:.4f}')
    print(f'model: {args.output}/{args.target}.pt')

if __name__ == '__main__':
    main()
