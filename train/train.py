#!/usr/bin/env python3
# train the stem separator model
#
# usage: python3 train.py --data /path/to/stems --size small --epochs 200
#
# expects data from batch_separate.py

import argparse, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import make_model
from dataset import StemDataset

def multi_resolution_stft_loss(pred, target, fft_sizes=[512, 1024, 2048]):
    """loss in frequency domain at multiple resolutions — catches both
    fine detail and broad spectral shape"""
    loss = 0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)

        # flatten to (batch * sources * channels, samples)
        p = pred.reshape(-1, pred.shape[-1])
        t = target.reshape(-1, target.shape[-1])

        pred_spec = torch.stft(p, n_fft, hop, window=window, return_complex=True).abs()
        target_spec = torch.stft(t, n_fft, hop, window=window, return_complex=True).abs()

        # L1 on magnitude + log magnitude
        loss += (pred_spec - target_spec).abs().mean()
        loss += (pred_spec.clamp(min=1e-7).log() - target_spec.clamp(min=1e-7).log()).abs().mean()

    return loss / len(fft_sizes)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    model = make_model(args.size)
    model = model.to(device)

    dataset = StemDataset(args.data, chunk_length=args.chunk_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # cosine decay with warmup
    warmup_steps = len(loader) * 5  # 5 epochs warmup
    total_steps = len(loader) * args.epochs
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    os.makedirs(args.output, exist_ok=True)
    best_loss = float('inf')

    print(f'  training for {args.epochs} epochs')
    print(f'  batch size: {args.batch_size}')
    print(f'  chunks: {args.chunk_length}s')
    print()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        t0 = time.time()

        for mix, stems in loader:
            mix = mix.to(device)
            stems = stems.to(device)

            pred = model(mix)  # (B, n_sources, channels, samples)

            # L1 in time domain
            l1_loss = (pred - stems).abs().mean()

            # multi-resolution STFT loss
            stft_loss = multi_resolution_stft_loss(pred, stems)

            loss = l1_loss + stft_loss

            optimizer.zero_grad()
            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f'  epoch {epoch+1:3d}/{args.epochs}  loss: {avg_loss:.4f}  lr: {lr:.6f}  time: {elapsed:.0f}s')

        # save best + periodic checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output, 'best.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss': best_loss,
                'config': {
                    'size': args.size,
                    'source_names': model.source_names,
                    'sample_rate': model.sample_rate,
                    'n_fft': model.n_fft,
                    'hop_length': model.hop_length,
                },
            }, save_path)
            print(f'    saved best model (loss: {best_loss:.4f})')

        if (epoch + 1) % 50 == 0:
            save_path = os.path.join(args.output, f'checkpoint_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
                'config': {
                    'size': args.size,
                    'source_names': model.source_names,
                    'sample_rate': model.sample_rate,
                    'n_fft': model.n_fft,
                    'hop_length': model.hop_length,
                },
            }, save_path)

    print(f'\ndone. best loss: {best_loss:.4f}')
    print(f'model saved to {args.output}/best.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to separated stems')
    parser.add_argument('--output', default='./checkpoints', help='where to save models')
    parser.add_argument('--size', default='small', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--chunk_length', type=int, default=10, help='seconds per training chunk')
    args = parser.parse_args()

    train(args)
