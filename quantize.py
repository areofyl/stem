#!/usr/bin/env python3
"""
Export and quantize htdemucs_ft for faster CPU inference.

Approaches:
1. Single model from bag (4x faster, slight quality drop)
2. PyTorch dynamic quantization (quantize linear/conv layers to int8)
3. Both combined

Usage: python3 quantize.py
  Outputs: ~/.cache/auris/htdemucs_ft_q.pt
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np

def get_model():
    import demucs.pretrained
    return demucs.pretrained.get_model("htdemucs_ft")

def benchmark(separate_fn, audio, sr, label, runs=1):
    """Benchmark a separation function."""
    print(f"\n  benchmarking: {label}")
    times = []
    for i in range(runs):
        start = time.time()
        result = separate_fn(audio)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    run {i+1}: {elapsed:.1f}s")
    avg = sum(times) / len(times)
    print(f"    average: {avg:.1f}s")
    return avg, result

def separate_with_demucs(model, audio, sr=44100):
    """Run demucs separation using its own apply module."""
    from demucs.apply import apply_model
    ref = audio.mean(0)
    audio = (audio - ref.mean()) / ref.std()
    sources = apply_model(model, audio[None], progress=False)[0]
    sources = sources * ref.std() + ref.mean()
    return sources

def main():
    print("--- loading model ---")
    bag = get_model()
    sr = bag.samplerate

    # Create 10s test signal
    print("--- creating test signal ---")
    test_audio = torch.randn(2, sr * 10)

    # Approach 1: Full bag of 4 (baseline)
    t_full, _ = benchmark(
        lambda a: separate_with_demucs(bag, a, sr),
        test_audio, sr, "full bag of 4 (baseline)"
    )

    # Approach 2: Single model from bag
    single = bag.models[0]
    single.eval()
    t_single, _ = benchmark(
        lambda a: separate_with_demucs(single, a, sr),
        test_audio, sr, "single model (1 of 4)"
    )

    # Approach 3: Single model + dynamic quantization
    single_q = torch.quantization.quantize_dynamic(
        bag.models[0],
        {nn.Linear},
        dtype=torch.qint8,
    )
    single_q.eval()
    t_quant_linear, _ = benchmark(
        lambda a: separate_with_demucs(single_q, a, sr),
        test_audio, sr, "single model + quantized linear"
    )

    # Approach 4: Single model + quantize conv + linear
    single_q2 = torch.quantization.quantize_dynamic(
        bag.models[1],  # use a fresh model from the bag
        {nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d},
        dtype=torch.qint8,
    )
    single_q2.eval()
    t_quant_all, _ = benchmark(
        lambda a: separate_with_demucs(single_q2, a, sr),
        test_audio, sr, "single model + quantized all layers"
    )

    print("\n--- results (10s audio) ---")
    print(f"  full bag (baseline):     {t_full:.1f}s")
    print(f"  single model:            {t_single:.1f}s  ({t_full/t_single:.1f}x faster)")
    print(f"  single + quant linear:   {t_quant_linear:.1f}s  ({t_full/t_quant_linear:.1f}x faster)")
    print(f"  single + quant all:      {t_quant_all:.1f}s  ({t_full/t_quant_all:.1f}x faster)")

    # Save the best quantized model
    cache_dir = os.path.expanduser("~/.cache/auris")
    os.makedirs(cache_dir, exist_ok=True)

    best_model = single_q2 if t_quant_all < t_quant_linear else single_q
    best_path = os.path.join(cache_dir, "htdemucs_ft_q.pt")

    # Save model state + config needed for reconstruction
    torch.save({
        "state_dict": best_model.state_dict(),
        "sources": bag.sources,
        "samplerate": sr,
    }, best_path)
    print(f"\n  saved quantized model to {best_path}")
    size_mb = os.path.getsize(best_path) / 1e6
    print(f"  size: {size_mb:.0f}MB")

if __name__ == "__main__":
    main()
