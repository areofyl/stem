I discovered spatial audio recently (in the last month) and it struck me as odd that all music files don't have it. I mean, once you hear it, there's no going back - it's just too good! 

I wanted to figure out a way to make stereo better (so that it doesn't sound like sound jumping between ears), so I made a program (it was not simple, I will tell you that much!) which would rotate a sound around your head with stereo using [HRTF](https://en.wikipedia.org/wiki/Head-related_transfer_function), which worked *really* well.

I finally felt like stereo could be heard with simple, lossy files which I downloaded with yt-dlp. I didn't need the fancy speakers and vinyls (ehem audiophiles). I just need some opus files, C, and math :)

This program is an extended version of that project, where it uses htdemucs to derive all instrument tracks from an audio file, then places sounds in different locations around the room, and uses HRTF to make it seem more immersive. And there's also some reverb stuff to make it sound like sound can also bounce off walls, but that's less refined. I've tried it on some songs with this program and I heard multiple keyboard lines from different directions that I'd been completely unaware of when listening to it without the program.

In fast mode it takes about 0.7x the length of the file for the AI to break it down on CPU, plus a couple seconds for the HRTF and spatializing. So a 4 minute song takes about 3 minutes. Quality mode (-q) runs all 4 models and takes about 4x longer. On a GPU it would be **way** faster but I'm running this on CPU.

I will be training a model to be better than demucs (speed-wise) (check out [the training dir](github.com/areofyl/stem/train/)).

## Usage

```
./stem song.mp3
```

That's it. Outputs `song_3d.wav` and saves the isolated stems to `stemmed-song/`.

```
./stem -t 30 song.mp3              # first 30 seconds only (-t is trim)
./stem -t 0 song.mp3               # full track
./stem -q song.mp3                 # quality mode (runs all 4 models, ~4x slower)
./stem --theater song.mp3          # 6-stem theater layout (guitar and piano get their own positions, bass from all directions)
./stem --glue 50 song.mp3          # blend 50% of the original mix back in (good for already-immersive/well-produced songs)
```

## Building

```
gcc -O2 -o spatialize spatialize.c -lsndfile -lm
pip install demucs
```

Listen with headphones. You should be able to hear sound coming from a direction *and* height!
