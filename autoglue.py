#!/usr/bin/env python3
# analyze separation quality, output a glue percentage (0-60)
# compares sum-of-stems to original mix
# usage: python3 autoglue.py original.wav stems_dir/

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

SOURCES = ['drums', 'bass', 'other', 'vocals']

def bandpass_fft(signal, sr, low, high):
    """quick bandpass using FFT, no scipy needed"""
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    spec[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spec, n=len(signal))

def main():
    if len(sys.argv) < 3:
        print('0')
        sys.exit(0)

    orig, sr = sf.read(sys.argv[1], dtype='float32', always_2d=True)
    stem_dir = Path(sys.argv[2])

    mixed = np.zeros_like(orig)
    for s in SOURCES:
        p = stem_dir / f'{s}.wav'
        if p.exists():
            audio, _ = sf.read(str(p), dtype='float32', always_2d=True)
            min_len = min(len(mixed), len(audio))
            mixed[:min_len] += audio[:min_len]

    min_len = min(len(orig), len(mixed))
    orig = orig[:min_len]
    mixed = mixed[:min_len]

    error = orig - mixed
    signal_power = np.mean(orig ** 2)
    error_power = np.mean(error ** 2)

    if error_power < 1e-10:
        print('0')
        return

    ser_db = 10 * np.log10(signal_power / error_power)

    # vocal band error (200-4000Hz) weighs more since vocal artifacts are most noticeable
    orig_vocal = bandpass_fft(orig[:, 0], sr, 200, 4000)
    mixed_vocal = bandpass_fft(mixed[:, 0], sr, 200, 4000)
    vocal_error = orig_vocal - mixed_vocal
    vocal_ser = 10 * np.log10(np.mean(orig_vocal ** 2) / max(np.mean(vocal_error ** 2), 1e-10))

    combined_ser = 0.4 * ser_db + 0.6 * vocal_ser

    if combined_ser > 25:
        glue = 0
    elif combined_ser > 18:
        glue = 10
    elif combined_ser > 14:
        glue = 20
    elif combined_ser > 10:
        glue = 30
    elif combined_ser > 7:
        glue = 40
    else:
        glue = 55

    print(int(glue))

if __name__ == '__main__':
    main()
