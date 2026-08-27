Training your own stem separation model. Band-split transformer that chops the spectrogram into mel-spaced frequency bands, runs interleaved band/time attention, and outputs bounded masks. Processes stereo jointly.

You'll want a cloud GPU for this (A100 on Lambda/RunPod, ~$1.50/hr). The whole thing takes maybe $10-15 and an afternoon.

Data:
  Best results come from mixing real stems with pseudo-labeled ones.
  1. python3 get_musdb.py /path/to/stems
     Downloads MUSDB18-HQ, 150 songs with real studio stems, ~15GB.
  2. python3 batch_separate.py /path/to/your/music /path/to/stems
     Runs htdemucs_6s on your library, adds pseudo-labeled stems on top.

  The dataset loader does remix augmentation. At each step it can sample each stem from a different song, giving N^4 combinations from N songs. 500 songs = 62 billion unique training mixes from the same data.

Training:
  3. python3 train.py --data /path/to/stems --size small --epochs 200 --batch_size 64
     Uses bfloat16 + torch.compile on A100, trains in 4-8hrs.
  4. python3 inference.py checkpoints/best.pt song.wav output/
     Test it.
  5. Download best.pt (~50MB), use it in stem.

Model sizes:
  tiny   ~2.3M params   runs in <1s, rough but fast
  small  ~3.7M params   good starting point
  medium ~7.5M params   best quality, still fast on CPU

The loss function weights bass errors 2.5x heavier and boosts sub-250Hz penalties because low-frequency bleed is the most perceptible artifact when you spatialize the output.

Train small first, listen, then decide if you need to go bigger.
