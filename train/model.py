#!/usr/bin/env python3
# stem separator model
# band-split transformer loosely based on BSRoformer
#
# key ideas:
#   - STFT both channels jointly (stereo matters for panning info)
#   - split freq axis into mel-spaced bands (more resolution in lows)
#   - alternate band-attention and time-attention layers (interleaved)
#   - bounded masks via sigmoid (prevents training blowups)
#   - skip connections everywhere
#   - overlap-add chunking for inference on full songs

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def mel_band_edges(n_bands, n_fft, sr):
    """compute band edges spaced according to mel scale.
    more bands in the lows where human hearing has finer resolution."""
    freq_bins = n_fft // 2 + 1
    max_freq = sr / 2

    # mel scale boundaries
    mel_low = 2595 * math.log10(1 + 20 / 700)
    mel_high = 2595 * math.log10(1 + max_freq / 700)
    mel_edges = np.linspace(mel_low, mel_high, n_bands + 1)
    hz_edges = 700 * (10 ** (mel_edges / 2595) - 1)

    # convert hz to bin indices
    bin_edges = (hz_edges / max_freq * (freq_bins - 1)).astype(int)
    bin_edges[0] = 0
    bin_edges[-1] = freq_bins

    # build (start, end) pairs, merge any empty bands
    bands = []
    for i in range(n_bands):
        start, end = int(bin_edges[i]), int(bin_edges[i + 1])
        if end <= start:
            end = start + 1
        bands.append((start, end))

    return bands


class BandSplitModule(nn.Module):
    """split stereo spectrogram into mel-spaced bands, project to embeddings"""

    def __init__(self, n_fft, n_bands, emb_dim, sr, n_channels=2):
        super().__init__()
        self.bands = mel_band_edges(n_bands, n_fft, sr)
        self.n_bands = len(self.bands)

        # input: real + imag, both channels = 4x band_size
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear((end - start) * n_channels * 2, emb_dim),
                nn.GELU(),
            )
            for start, end in self.bands
        ])

    def forward(self, spec):
        # spec: (B, channels, 2, freq, time) — channels=2 (stereo), 2=real/imag
        B, C, RI, F, T = spec.shape

        band_features = []
        for i, (start, end) in enumerate(self.bands):
            # grab band across all channels and real/imag
            band = spec[:, :, :, start:end, :]  # (B, C, 2, band_size, T)
            band = band.permute(0, 4, 1, 2, 3).reshape(B, T, -1)  # (B, T, C*2*band_size)
            band = self.projections[i](band)  # (B, T, emb_dim)
            band_features.append(band)

        return torch.stack(band_features, dim=1)  # (B, n_bands, T, emb_dim)


class InterleavedAttention(nn.Module):
    """alternating band-attention and time-attention layers with skip connections.
    this is the core — lets the model see across frequencies AND across time."""

    def __init__(self, emb_dim, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                # band attention: at each time step, look across all frequency bands
                'band_norm': nn.LayerNorm(emb_dim),
                'band_attn': nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True),
                'band_ff': nn.Sequential(
                    nn.Linear(emb_dim, emb_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(emb_dim * 2, emb_dim),
                    nn.Dropout(dropout),
                ),
                'band_ff_norm': nn.LayerNorm(emb_dim),
                # time attention: for each band, look across all time steps
                'time_norm': nn.LayerNorm(emb_dim),
                'time_attn': nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True),
                'time_ff': nn.Sequential(
                    nn.Linear(emb_dim, emb_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(emb_dim * 2, emb_dim),
                    nn.Dropout(dropout),
                ),
                'time_ff_norm': nn.LayerNorm(emb_dim),
            }))

    def forward(self, x):
        # x: (B, n_bands, T, emb_dim)
        B, N, T, D = x.shape

        for layer in self.layers:
            # band attention (skip connection)
            r = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
            r = layer['band_norm'](r)
            r = layer['band_attn'](r, r, r, need_weights=False)[0] + r
            r = r.reshape(B, T, N, D).permute(0, 2, 1, 3)
            x = x + r
            # band feedforward (skip connection)
            r = layer['band_ff_norm'](x)
            x = x + layer['band_ff'](r)

            # time attention (skip connection)
            r = x.reshape(B * N, T, D)
            r = layer['time_norm'](r)
            r = layer['time_attn'](r, r, r, need_weights=False)[0] + r
            r = r.reshape(B, N, T, D)
            x = x + r
            # time feedforward (skip connection)
            r = layer['time_ff_norm'](x)
            x = x + layer['time_ff'](r)

        return x


class BandMergeModule(nn.Module):
    """project embeddings back to stereo spectrogram masks (bounded by sigmoid)"""

    def __init__(self, n_fft, bands, emb_dim, n_sources, n_channels=2):
        super().__init__()
        self.n_sources = n_sources
        self.n_channels = n_channels
        self.bands = bands

        self.projections = nn.ModuleList([
            nn.Linear(emb_dim, (end - start) * n_channels * 2 * n_sources)
            for start, end in self.bands
        ])

    def forward(self, x, orig_freq):
        # x: (B, n_bands, T, emb_dim)
        B, N, T, D = x.shape
        C = self.n_channels

        output = torch.zeros(B, self.n_sources, C, 2, orig_freq, T, device=x.device)

        for i, (start, end) in enumerate(self.bands):
            band_size = end - start
            raw = self.projections[i](x[:, i])  # (B, T, band_size * C * 2 * n_sources)
            raw = raw.reshape(B, T, self.n_sources, C, 2, band_size)
            raw = raw.permute(0, 2, 3, 4, 5, 1)  # (B, sources, C, 2, band_size, T)
            output[:, :, :, :, start:end, :] = raw

        # sigmoid to bound masks between 0 and 1
        return torch.sigmoid(output)


class StemSeparator(nn.Module):
    """
    Band-split transformer for stem separation.

    Processes stereo jointly, uses mel-spaced bands, interleaved
    band/time attention, bounded masks, skip connections.
    """

    def __init__(
        self,
        n_fft=2048,
        hop_length=512,
        n_bands=16,
        emb_dim=128,
        n_heads=4,
        n_layers=4,
        n_sources=4,
        source_names=None,
        sample_rate=44100,
        chunk_seconds=10,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_sources = n_sources
        self.source_names = source_names or ['drums', 'bass', 'other', 'vocals']
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_seconds * sample_rate
        self.overlap_samples = sample_rate  # 1 second overlap for chunked inference

        self.band_split = BandSplitModule(n_fft, n_bands, emb_dim, sample_rate)
        self.attention = InterleavedAttention(emb_dim, n_heads, n_layers)
        self.band_merge = BandMergeModule(n_fft, self.band_split.bands, emb_dim, n_sources)

        self.window = None

    def _get_window(self, device):
        if self.window is None or self.window.device != device:
            self.window = torch.hann_window(self.n_fft, device=device)
        return self.window

    def stft(self, x):
        # x: (B, samples) mono
        return torch.stft(x, self.n_fft, self.hop_length,
                         window=self._get_window(x.device), return_complex=True)

    def istft(self, x, length):
        return torch.istft(x, self.n_fft, self.hop_length,
                          window=self._get_window(x.device), length=length)

    def forward_chunk(self, mix):
        """process a single chunk. mix: (B, 2, samples)"""
        B, C, L = mix.shape

        # STFT both channels
        specs = []
        for ch in range(C):
            spec = self.stft(mix[:, ch])  # (B, freq, time) complex
            specs.append(spec)
        # stack: (B, 2, freq, time)
        spec_stack = torch.stack(specs, dim=1)
        freq = spec_stack.shape[2]

        # separate real and imag: (B, channels, 2, freq, time)
        spec_ri = torch.stack([spec_stack.real, spec_stack.imag], dim=2)

        # band split -> interleaved attention -> band merge
        bands = self.band_split(spec_ri)
        bands = self.attention(bands)
        masks = self.band_merge(bands, freq)  # (B, sources, channels, 2, freq, time)

        # apply masks to original spec
        sources = []
        for s in range(self.n_sources):
            source_channels = []
            for ch in range(C):
                mask_r = masks[:, s, ch, 0]  # (B, freq, time)
                mask_i = masks[:, s, ch, 1]
                # apply mask: multiply magnitude, keep phase from original
                orig = specs[ch]
                masked = orig * torch.complex(mask_r, mask_i)
                audio = self.istft(masked, L)
                source_channels.append(audio)
            sources.append(torch.stack(source_channels, dim=1))  # (B, 2, samples)

        return torch.stack(sources, dim=1)  # (B, n_sources, 2, samples)

    def forward(self, mix):
        """forward pass, handles arbitrary length via chunking during training"""
        return self.forward_chunk(mix)

    @torch.no_grad()
    def separate(self, mix_audio):
        """inference on full song with overlap-add chunking.
        takes (2, samples) tensor, returns (n_sources, 2, samples)"""
        if not isinstance(mix_audio, torch.Tensor):
            mix_audio = torch.from_numpy(mix_audio).float()
        if mix_audio.dim() == 1:
            mix_audio = mix_audio.unsqueeze(0).repeat(2, 1)

        C, L = mix_audio.shape
        chunk = self.chunk_samples
        overlap = self.overlap_samples

        # short enough to process in one go
        if L <= chunk:
            return self.forward_chunk(mix_audio.unsqueeze(0))[0]

        # overlap-add for long files
        stride = chunk - overlap
        output = torch.zeros(self.n_sources, C, L, device=mix_audio.device)
        weight = torch.zeros(L, device=mix_audio.device)

        # triangular window for crossfading overlapping chunks
        win = torch.ones(chunk, device=mix_audio.device)
        ramp = torch.linspace(0, 1, overlap, device=mix_audio.device)
        win[:overlap] = ramp
        win[-overlap:] = ramp.flip(0)

        pos = 0
        while pos < L:
            end = min(pos + chunk, L)
            start = end - chunk if end == L and end - pos < chunk else pos
            actual_len = end - start

            chunk_audio = mix_audio[:, start:end]
            if chunk_audio.shape[1] < chunk:
                # pad last chunk
                pad = chunk - chunk_audio.shape[1]
                chunk_audio = F.pad(chunk_audio, (0, pad))

            result = self.forward_chunk(chunk_audio.unsqueeze(0))[0]  # (sources, 2, chunk)
            result = result[:, :, :actual_len]

            w = win[:actual_len]
            output[:, :, start:end] += result * w.unsqueeze(0).unsqueeze(0)
            weight[start:end] += w

            pos += stride
            if end == L:
                break

        # normalize by overlap weights
        weight = weight.clamp(min=1e-8)
        output = output / weight.unsqueeze(0).unsqueeze(0)

        return output


def make_model(size='small'):
    """create a model. sizes: tiny (~3M), small (~8M), medium (~15M)"""
    configs = {
        'tiny': dict(n_bands=12, emb_dim=96,  n_heads=4, n_layers=2),
        'small': dict(n_bands=20, emb_dim=128, n_heads=4, n_layers=4),
        'medium': dict(n_bands=24, emb_dim=192, n_heads=6, n_layers=6),
    }
    config = configs[size]
    model = StemSeparator(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  model size: {size} ({n_params/1e6:.1f}M params)')
    return model
