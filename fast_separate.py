#!/usr/bin/env python3
# fast demucs: grabs one model from the bag instead of running all 4
# usage: fast_separate.py input.wav output_dir

import sys, os
import torch
import soundfile as sf

def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} input.wav output_dir", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    model_name = os.environ.get("STEM_MODEL", "htdemucs_ft")

    import demucs.pretrained
    from demucs.apply import apply_model
    from demucs.audio import save_audio

    print(f"  loading {model_name}...")
    bag = demucs.pretrained.get_model(model_name)

    # pull one model out of the bag for ~4x speedup
    if hasattr(bag, 'models') and len(bag.models) > 1:
        model = bag.models[0]
        print(f"  using 1/{len(bag.models)} models")
    else:
        model = bag

    model.eval()
    sr = bag.samplerate

    # load and make stereo
    audio, _ = sf.read(input_path, dtype='float32', always_2d=True)
    audio = torch.from_numpy(audio.T)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)

    # normalize, separate, denormalize
    ref = audio.mean(0)
    mean, std = ref.mean(), ref.std()
    audio = (audio - mean) / std

    print(f"  separating...")
    with torch.no_grad():
        sources = apply_model(model, audio[None], progress=True)[0]
    sources = sources * std + mean

    for i, name in enumerate(bag.sources):
        save_audio(sources[i], os.path.join(output_dir, f"{name}.wav"), sr)
        print(f"  saved {name}")

    print(f"  done: {len(bag.sources)} stems")

if __name__ == "__main__":
    main()
