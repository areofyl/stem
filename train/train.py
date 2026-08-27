#!/usr/bin/env python3
# train the stem separator
#
# python3 train.py --data /path/to/stems --size small --epochs 200
# python3 train.py --data /path/to/stems --size small --epochs 200 --batch_size 64

import argparse, os, time, math
import torch
from torch.utils.data import DataLoader
from model import make_model
from dataset import StemDataset, train_val_split, SOURCES

# stem indices for weighted loss
BASS_IDX = SOURCES.index('bass')


def multi_resolution_stft_loss(pred, target, fft_sizes=[512, 1024, 2048],
                                sr=44100, bass_weight=2.5, low_freq_boost=2.0):
    """multi-res STFT loss with per-stem weighting and low-frequency emphasis"""
    B, S, C, L = pred.shape  # batch, sources, channels, samples

    total_loss = 0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)
        freq_bins = n_fft // 2 + 1

        # frequency weights: boost sub-250Hz
        freqs = torch.linspace(0, sr / 2, freq_bins, device=pred.device)
        freq_w = torch.ones(freq_bins, device=pred.device)
        freq_w[freqs < 250] = low_freq_boost

        # per-stem weights
        stem_weights = torch.ones(S, device=pred.device)
        stem_weights[BASS_IDX] = bass_weight

        for s in range(S):
            p = pred[:, s].reshape(-1, L)  # (B*C, samples)
            t = target[:, s].reshape(-1, L)

            p_mag = torch.stft(p, n_fft, hop, window=window, return_complex=True).abs()
            t_mag = torch.stft(t, n_fft, hop, window=window, return_complex=True).abs()

            # frequency-weighted L1 on magnitude
            diff = (p_mag - t_mag).abs() * freq_w.unsqueeze(0).unsqueeze(-1)
            mag_loss = diff.mean()

            # frequency-weighted L1 on log magnitude
            log_diff = (p_mag.clamp(min=1e-7).log() - t_mag.clamp(min=1e-7).log()).abs()
            log_loss = (log_diff * freq_w.unsqueeze(0).unsqueeze(-1)).mean()

            total_loss += (mag_loss + log_loss) * stem_weights[s]

    return total_loss / (len(fft_sizes) * S)


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for mix, stems in val_loader:
            mix, stems = mix.to(device), stems.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == 'cuda')):
                pred = model(mix)
                loss = multi_resolution_stft_loss(pred, stems)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'
    n_workers = min(os.cpu_count() // 2, 16) if os.cpu_count() else 4
    print(f'device: {device}')
    if use_amp:
        print(f'  bfloat16 mixed precision')

    model = make_model(args.size).to(device)

    # torch.compile for kernel fusion speedup (A100)
    if device == 'cuda' and hasattr(torch, 'compile'):
        print('  compiling model...')
        model = torch.compile(model)

    train_songs, val_songs = train_val_split(args.data)
    print(f'  train: {len(train_songs)} songs, val: {len(val_songs)} songs')

    train_set = StemDataset(args.data, chunk_length=args.chunk_length, songs=train_songs, augment=True)
    val_set = StemDataset(args.data, chunk_length=args.chunk_length, songs=val_songs, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=n_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=max(n_workers // 2, 1), pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.GradScaler(device, enabled=use_amp)

    # cosine schedule with warmup
    warmup_steps = len(train_loader) * 5
    total_steps = len(train_loader) * args.epochs
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    # resume
    start_epoch = 0
    best_val_loss = float('inf')
    resume_path = os.path.join(args.output, 'latest.pt')
    if os.path.exists(resume_path):
        print(f'  resuming from {resume_path}')
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

    os.makedirs(args.output, exist_ok=True)

    print(f'  epochs: {start_epoch} -> {args.epochs}')
    print(f'  batch size: {args.batch_size}')
    print(f'  workers: {n_workers}')
    print(f'  chunks: {args.chunk_length}s')
    print()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        t0 = time.time()

        for mix, stems in train_loader:
            mix, stems = mix.to(device), stems.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                pred = model(mix)
                loss = multi_resolution_stft_loss(pred, stems)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        val_loss = validate(model, val_loader, device)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f'  epoch {epoch+1:3d}/{args.epochs}  '
              f'train: {train_loss:.4f}  val: {val_loss:.4f}  '
              f'lr: {lr:.6f}  time: {elapsed:.0f}s')

        config = {
            'size': args.size,
            'source_names': SOURCES,
            'sample_rate': 44100,
            'n_fft': 2048,
            'hop_length': 512,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'loss': val_loss, 'config': config,
            }, os.path.join(args.output, 'best.pt'))
            print(f'    new best val loss: {best_val_loss:.4f}')

        torch.save({
            'epoch': epoch, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss, 'config': config,
        }, os.path.join(args.output, 'latest.pt'))

        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': val_loss, 'config': config,
            }, os.path.join(args.output, f'checkpoint_{epoch+1}.pt'))

    print(f'\ndone. best val loss: {best_val_loss:.4f}')
    print(f'model: {args.output}/best.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='./checkpoints')
    parser.add_argument('--size', default='small', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--chunk_length', type=int, default=10)
    args = parser.parse_args()

    train(args)
