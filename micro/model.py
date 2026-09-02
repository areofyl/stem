#!/usr/bin/env python3
# tiny single-instrument separator (~300K params)
# RNNoise-inspired: GRU + conv layers, processes frame-by-frame
# no attention, no FFT-domain, just raw waveform in -> waveform out
#
# each model extracts ONE instrument from a mix

import torch
import torch.nn as nn

class MicroSeparator(nn.Module):
    """
    tiny separator for one instrument.
    processes audio in small frames (10ms) for real-time use.

    architecture:
      1. 1D conv encoder (mix waveform -> features)
      2. GRU (temporal context across frames)
      3. 1D conv decoder (features -> mask)
      4. multiply mask with input -> isolated instrument
    """

    def __init__(self, n_fft=512, hop=256, hidden=128, n_gru_layers=2):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        n_bins = n_fft // 2 + 1

        # encode magnitude spectrogram
        self.encoder = nn.Sequential(
            nn.Linear(n_bins, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # temporal modeling
        self.gru = nn.GRU(hidden, hidden, num_layers=n_gru_layers, batch_first=True)

        # decode to mask
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_bins),
            nn.Sigmoid(),  # mask between 0 and 1
        )

        self.window = None

    def _get_window(self, device):
        if self.window is None or self.window.device != device:
            self.window = torch.hann_window(self.n_fft, device=device)
        return self.window

    def forward(self, mix):
        """mix: (B, samples) mono. returns: (B, samples) isolated instrument"""
        B, L = mix.shape
        win = self._get_window(mix.device)

        # STFT
        spec = torch.stft(mix, self.n_fft, self.hop, window=win, return_complex=True)
        mag = spec.abs()     # (B, n_bins, T)
        phase = spec.angle()

        # (B, n_bins, T) -> (B, T, n_bins) for frame-by-frame processing
        mag_t = mag.permute(0, 2, 1)

        # encode -> GRU -> decode mask
        feat = self.encoder(mag_t)
        feat, _ = self.gru(feat)
        mask = self.decoder(feat)  # (B, T, n_bins)

        # apply mask to magnitude, keep original phase
        mask_t = mask.permute(0, 2, 1)  # (B, n_bins, T)
        out_spec = (mag * mask_t) * torch.exp(1j * phase)

        # ISTFT
        out = torch.istft(out_spec, self.n_fft, self.hop, window=win, length=L)
        return out

    @torch.no_grad()
    def separate(self, audio):
        """convenience: takes (samples,) or (channels, samples), returns mono output"""
        if audio.dim() == 2:
            audio = audio.mean(0)  # stereo to mono
        return self.forward(audio.unsqueeze(0))[0]


def make_model():
    m = MicroSeparator()
    n = sum(p.numel() for p in m.parameters())
    print(f'  micro model: {n/1e3:.0f}K params')
    return m
