#!/usr/bin/env python3
# analyze how well the separation went and recommend a glue percentage
# compares sum-of-stems to original mix
# high error = bad separation = more glue needed
#
# usage: python3 autoglue.py original.wav stems_dir/
# prints a number 0-60

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

SOURCES = ['drums', 'bass', 'other', 'vocals']

def main():
    if len(sys.argv) < 3:
        print('0')
        sys.exit(0)

    orig_path = sys.argv[1]
    stem_dir = Path(sys.argv[2])

    orig, sr = sf.read(orig_path, dtype='float32', always_2d=True)

    # sum all stems
    mixed = np.zeros_like(orig)
    for s in SOURCES:
        p = stem_dir / f'{s}.wav'
        if p.exists():
            audio, _ = sf.read(str(p), dtype='float32', always_2d=True)
            min_len = min(len(mixed), len(audio))
            mixed[:min_len] += audio[:min_len]

    # trim to same length
    min_len = min(len(orig), len(mixed))
    orig = orig[:min_len]
    mixed = mixed[:min_len]

    # measure how different the reconstruction is from the original
    # using signal-to-error ratio in dB
    error = orig - mixed
    signal_power = np.mean(orig ** 2)
    error_power = np.mean(error ** 2)

    if error_power < 1e-10:
        # perfect reconstruction, no glue needed
        print('0')
        return

    ser_db = 10 * np.log10(signal_power / error_power)

    # also check per-band error (vocals live in 200-4000Hz mostly)
    # high error in the vocal range means worse perceived quality
    from scipy.signal import butter, sosfilt

    vocal_sos = butter(4, [200, 4000], btype='band', fs=sr, output='sos')
    orig_vocal = sosfilt(vocal_sos, orig[:, 0])
    mixed_vocal = sosfilt(vocal_sos, mixed[:, 0])
    vocal_error = orig_vocal - mixed_vocal
    vocal_ser = 10 * np.log10(np.mean(orig_vocal ** 2) / max(np.mean(vocal_error ** 2), 1e-10))

    # map SER to glue percentage
    # high SER (>20dB) = good separation = low glue
    # low SER (<8dB) = bad separation = high glue
    # vocal band weighs more because vocal artifacts are most noticeable
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
