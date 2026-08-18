#!/usr/bin/env python3
# lightweight stem separator model
# band-split architecture inspired by BSRoformer but way smaller
#
# idea: split spectrogram into frequency bands, process each with
# a small transformer, recombine. simple and effective.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BandSplitModule(nn.Module):
    """split spectrogram into frequency bands and project each to embedding dim"""

    def __init__(self, n_fft, n_bands, emb_dim):
        super().__init__()
        self.n_bands = n_bands

        # divide frequency bins into roughly equal bands
        freq_bins = n_fft // 2 + 1
        band_size = freq_bins // n_bands
        self.bands = []
        for i in range(n_bands):
            start = i * band_size
            end = start + band_size if i < n_bands - 1 else freq_bins
            self.bands.append((start, end))

        # one linear projection per band (real + imag = 2x)
        self.projections = nn.ModuleList([
            nn.Linear((end - start) * 2, emb_dim)
            for start, end in self.bands
        ])

    def forward(self, spec):
        # spec: (batch, 2, freq, time) — real and imag stacked on dim 1
        batch, _, freq, time = spec.shape

        band_features = []
        for i, (start, end) in enumerate(self.bands):
            # grab this band from both real and imag, flatten
            band = spec[:, :, start:end, :]  # (B, 2, band_size, T)
            band = band.permute(0, 3, 1, 2).reshape(batch, time, -1)  # (B, T, 2*band_size)
            band = self.projections[i](band)  # (B, T, emb_dim)
            band_features.append(band)

        # stack bands: (B, n_bands, T, emb_dim)
        return torch.stack(band_features, dim=1)


class BandTransformer(nn.Module):
    """small transformer that processes each time step across bands"""

    def __init__(self, emb_dim, n_heads, n_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 2,  # keep it small
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x: (B, n_bands, T, emb_dim)
        B, N, T, D = x.shape

        # reshape to process all time steps: treat (B*T) as batch, N as sequence
        x = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.reshape(B, T, N, D).permute(0, 2, 1, 3)

        return x  # (B, n_bands, T, emb_dim)


class TimeTransformer(nn.Module):
    """process each band across time"""

    def __init__(self, emb_dim, n_heads, n_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x: (B, n_bands, T, emb_dim)
        B, N, T, D = x.shape

        # process each band independently across time
        x = x.reshape(B * N, T, D)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.reshape(B, N, T, D)

        return x


class BandMergeModule(nn.Module):
    """project band embeddings back to spectrogram"""

    def __init__(self, n_fft, n_bands, emb_dim, n_sources):
        super().__init__()
        self.n_sources = n_sources

        freq_bins = n_fft // 2 + 1
        band_size = freq_bins // n_bands
        self.bands = []
        for i in range(n_bands):
            start = i * band_size
            end = start + band_size if i < n_bands - 1 else freq_bins
            self.bands.append((start, end))

        # project back to freq domain (output real + imag for each source)
        self.projections = nn.ModuleList([
            nn.Linear(emb_dim, (end - start) * 2 * n_sources)
            for start, end in self.bands
        ])

    def forward(self, x, orig_freq):
        # x: (B, n_bands, T, emb_dim)
        B, N, T, D = x.shape

        # reconstruct spectrogram masks for each source
        output = torch.zeros(B, self.n_sources, 2, orig_freq, T, device=x.device)

        for i, (start, end) in enumerate(self.bands):
            band_size = end - start
            band = self.projections[i](x[:, i])  # (B, T, band_size * 2 * n_sources)
            band = band.reshape(B, T, self.n_sources, 2, band_size)
            band = band.permute(0, 2, 3, 4, 1)  # (B, n_sources, 2, band_size, T)
            output[:, :, :, start:end, :] = band

        return output


class AurisSeparator(nn.Module):
    """
    small band-split transformer for stem separation.

    default config: ~8M params
    processes audio as STFT, splits into frequency bands,
    runs band and time transformers, outputs source masks.
    """

    def __init__(
        self,
        n_fft=2048,
        hop_length=512,
        n_bands=16,
        emb_dim=128,
        n_heads=4,
        band_layers=2,
        time_layers=2,
        n_sources=4,
        source_names=None,
        sample_rate=44100,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_sources = n_sources
        self.source_names = source_names or ['drums', 'bass', 'other', 'vocals']
        self.sample_rate = sample_rate

        self.band_split = BandSplitModule(n_fft, n_bands, emb_dim)
        self.band_transformer = BandTransformer(emb_dim, n_heads, band_layers)
        self.time_transformer = TimeTransformer(emb_dim, n_heads, time_layers)
        self.band_merge = BandMergeModule(n_fft, n_bands, emb_dim, n_sources)

        self.window = None  # created on first forward pass

    def stft(self, x):
        if self.window is None or self.window.device != x.device:
            self.window = torch.hann_window(self.n_fft, device=x.device)
        return torch.stft(x, self.n_fft, self.hop_length, window=self.window, return_complex=True)

    def istft(self, x, length):
        if self.window is None or self.window.device != x.device:
            self.window = torch.hann_window(self.n_fft, device=x.device)
        return torch.istft(x, self.n_fft, self.hop_length, window=self.window, length=length)

    def forward(self, mix):
        # mix: (B, channels, samples) — stereo
        B, C, L = mix.shape

        # process each channel, average results (simple stereo handling)
        all_sources = []
        for ch in range(C):
            spec = self.stft(mix[:, ch])  # (B, freq, time) complex
            freq = spec.shape[1]
            time = spec.shape[2]

            # stack real and imag
            spec_ri = torch.stack([spec.real, spec.imag], dim=1)  # (B, 2, freq, time)

            # band split -> transformers -> band merge
            bands = self.band_split(spec_ri)
            bands = self.band_transformer(bands)
            bands = self.time_transformer(bands)
            masks = self.band_merge(bands, freq)  # (B, n_sources, 2, freq, time)

            # apply masks to original spec
            channel_sources = []
            for s in range(self.n_sources):
                mask_complex = torch.complex(masks[:, s, 0], masks[:, s, 1])
                source_spec = spec * mask_complex
                source_audio = self.istft(source_spec, L)
                channel_sources.append(source_audio)

            all_sources.append(torch.stack(channel_sources, dim=1))  # (B, n_sources, samples)

        # stack channels: (B, n_sources, channels, samples)
        sources = torch.stack(all_sources, dim=2)
        return sources

    def separate(self, mix_audio):
        """convenience method for inference — takes (channels, samples) numpy/tensor"""
        if not isinstance(mix_audio, torch.Tensor):
            mix_audio = torch.from_numpy(mix_audio).float()
        if mix_audio.dim() == 1:
            mix_audio = mix_audio.unsqueeze(0).repeat(2, 1)
        was_numpy = False

        with torch.no_grad():
            sources = self.forward(mix_audio.unsqueeze(0))[0]

        return sources  # (n_sources, channels, samples)


def make_model(size='small'):
    """create a model. sizes: tiny (~3M), small (~8M), medium (~15M)"""
    configs = {
        'tiny': dict(n_bands=12, emb_dim=96, n_heads=4, band_layers=1, time_layers=1),
        'small': dict(n_bands=16, emb_dim=128, n_heads=4, band_layers=2, time_layers=2),
        'medium': dict(n_bands=20, emb_dim=192, n_heads=6, band_layers=3, time_layers=3),
    }
    config = configs[size]
    model = AurisSeparator(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  model size: {size} ({n_params/1e6:.1f}M params)')
    return model
