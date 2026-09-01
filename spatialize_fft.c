/*
 * spatialize_fft.c — frequency-domain binaural spatializer using measured HRTFs
 *
 * instead of per-sample biquad chains, this does:
 *   1. chunk the input into overlapping blocks
 *   2. FFT each block
 *   3. multiply by the HRTF for the target position (one complex mul per bin)
 *   4. IFFT back to time domain
 *   5. overlap-add
 *
 * uses MIT KEMAR measurements embedded in hrtf_data.h
 * includes FDN reverb on the final mix (same as spatialize.c)
 *
 * Compile: gcc -O3 -march=native -ffast-math -o spatialize_fft spatialize_fft.c -lsndfile -lm -lpthread
 * Usage:   same as spatialize — ./spatialize_fft -o out.wav vocals.wav:0:0 drums.wav:160:15 ...
 */

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sndfile.h>

#include "hrtf/hrtf_data.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- simple radix-2 FFT (no external dependency) ---- */

typedef struct { float re, im; } cpx;

static void fft(cpx *buf, int n, int inverse)
{
	/* bit-reversal permutation */
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1)
			j ^= bit;
		j ^= bit;
		if (i < j) {
			cpx tmp = buf[i]; buf[i] = buf[j]; buf[j] = tmp;
		}
	}

	for (int len = 2; len <= n; len <<= 1) {
		float ang = 2.0f * (float)M_PI / len * (inverse ? -1 : 1);
		cpx wlen = { cosf(ang), sinf(ang) };
		for (int i = 0; i < n; i += len) {
			cpx w = { 1.0f, 0.0f };
			for (int j = 0; j < len / 2; j++) {
				cpx u = buf[i + j];
				cpx v = {
					buf[i + j + len/2].re * w.re - buf[i + j + len/2].im * w.im,
					buf[i + j + len/2].re * w.im + buf[i + j + len/2].im * w.re
				};
				buf[i + j] = (cpx){ u.re + v.re, u.im + v.im };
				buf[i + j + len/2] = (cpx){ u.re - v.re, u.im - v.im };
				float wr = w.re * wlen.re - w.im * wlen.im;
				w.im = w.re * wlen.im + w.im * wlen.re;
				w.re = wr;
			}
		}
	}

	if (inverse) {
		float scale = 1.0f / n;
		for (int i = 0; i < n; i++) {
			buf[i].re *= scale;
			buf[i].im *= scale;
		}
	}
}

/* real-signal FFT: input n floats, output n/2+1 complex bins */
static void rfft(float *in, cpx *out, int n)
{
	cpx *tmp = calloc(n, sizeof(cpx));
	for (int i = 0; i < n; i++)
		tmp[i] = (cpx){ in[i], 0.0f };
	fft(tmp, n, 0);
	for (int i = 0; i <= n/2; i++)
		out[i] = tmp[i];
	free(tmp);
}

/* inverse: n/2+1 complex bins to n floats */
static void irfft(cpx *in, float *out, int n)
{
	cpx *tmp = calloc(n, sizeof(cpx));
	for (int i = 0; i <= n/2; i++)
		tmp[i] = in[i];
	/* mirror conjugate */
	for (int i = 1; i < n/2; i++)
		tmp[n - i] = (cpx){ in[i].re, -in[i].im };
	fft(tmp, n, 1);
	for (int i = 0; i < n; i++)
		out[i] = tmp[i].re;
	free(tmp);
}

/* ---- HRTF lookup with interpolation ---- */

static int find_nearest_hrtf(float az_deg, float el_deg)
{
	/* find the closest measured position */
	float best_dist = 1e9f;
	int best = 0;

	for (int i = 0; i < HRTF_N_POSITIONS; i++) {
		float daz = hrtf_positions[i][0] - az_deg;
		float del = hrtf_positions[i][1] - el_deg;
		/* wrap azimuth */
		while (daz > 180) daz -= 360;
		while (daz < -180) daz += 360;
		float dist = daz * daz + del * del * 4.0f; /* weight elevation more */
		if (dist < best_dist) {
			best_dist = dist;
			best = i;
		}
	}
	return best;
}

static void get_hrtf(int pos_idx, cpx *left, cpx *right)
{
	/* load pre-FFT'd HRTF from the embedded data */
	const float *data_l = hrtf_data[pos_idx][0];
	const float *data_r = hrtf_data[pos_idx][1];
	for (int i = 0; i < HRTF_N_BINS; i++) {
		left[i]  = (cpx){ data_l[i*2], data_l[i*2+1] };
		right[i] = (cpx){ data_r[i*2], data_r[i*2+1] };
	}
}

/* ---- FDN reverb (same as spatialize.c) ---- */

#define FDN_N 8

typedef struct { float *buf; int len, pos; } FDNDelay;

typedef struct {
	FDNDelay delays[FDN_N];
	float feedback, damp_coeff;
	float damp[FDN_N];
	float gains_l[FDN_N], gains_r[FDN_N];
	float wet;
} Reverb;

static void reverb_init(Reverb *rv, double sr, float wet)
{
	double delay_ms[FDN_N] = { 19.3, 23.7, 29.1, 31.7, 37.3, 41.1, 43.9, 47.3 };
	rv->feedback = 0.82f;
	rv->damp_coeff = 0.35f;
	rv->wet = wet;
	for (int i = 0; i < FDN_N; i++) {
		int len = (int)(delay_ms[i] * sr / 1000.0);
		rv->delays[i].buf = calloc(len, sizeof(float));
		rv->delays[i].len = len;
		rv->delays[i].pos = 0;
		rv->damp[i] = 0.0f;
		float pan = (i % 2 == 0) ? 0.6f : 0.4f;
		pan += (i / 2) * 0.05f - 0.075f;
		rv->gains_l[i] = 1.0f - pan;
		rv->gains_r[i] = pan;
	}
}

static void reverb_process(Reverb *rv, float dry_l, float dry_r, float *out_l, float *out_r)
{
	float input = (dry_l + dry_r) * 0.5f;
	float taps[FDN_N];
	for (int i = 0; i < FDN_N; i++)
		taps[i] = rv->delays[i].buf[rv->delays[i].pos];

	float mixed[FDN_N];
	float scale = 1.0f / 2.828f;
	for (int i = 0; i < FDN_N; i++) {
		mixed[i] = 0.0f;
		for (int j = 0; j < FDN_N; j++) {
			int bits = i & j, parity = 0;
			while (bits) { parity ^= 1; bits &= bits - 1; }
			mixed[i] += parity ? -taps[j] : taps[j];
		}
		mixed[i] *= scale;
	}

	float wet_l = 0.0f, wet_r = 0.0f;
	for (int i = 0; i < FDN_N; i++) {
		float fb = mixed[i] * rv->feedback;
		rv->damp[i] = fb * (1.0f - rv->damp_coeff) + rv->damp[i] * rv->damp_coeff;
		rv->delays[i].buf[rv->delays[i].pos] = rv->damp[i] + input * 0.15f;
		rv->delays[i].pos = (rv->delays[i].pos + 1) % rv->delays[i].len;
		wet_l += taps[i] * rv->gains_l[i];
		wet_r += taps[i] * rv->gains_r[i];
	}
	wet_l /= FDN_N;
	wet_r /= FDN_N;
	*out_l = dry_l + wet_l * rv->wet;
	*out_r = dry_r + wet_r * rv->wet;
}

static void reverb_free(Reverb *rv)
{
	for (int i = 0; i < FDN_N; i++) free(rv->delays[i].buf);
}

/* ---- stem loading ---- */

typedef struct {
	char   *filename;
	float  az_deg, el_deg;
	float  stem_gain;
	float  *samples_l, *samples_r;
	int    num_frames;
	int    is_stereo;
	/* HRTF for this position */
	cpx    hrtf_l[HRTF_N_BINS];
	cpx    hrtf_r[HRTF_N_BINS];
	/* output buffer (filled by worker thread) */
	float  *out_l, *out_r;
	int    out_frames;
} Stem;

static int load_stem(Stem *s, int sr)
{
	SF_INFO info = {0};
	SNDFILE *sf = sf_open(s->filename, SFM_READ, &info);
	if (!sf) {
		fprintf(stderr, "Can't open %s: %s\n", s->filename, sf_strerror(NULL));
		return -1;
	}
	s->num_frames = (int)info.frames;
	s->samples_l = calloc(s->num_frames, sizeof(float));
	s->samples_r = calloc(s->num_frames, sizeof(float));

	if (info.channels == 1) {
		sf_readf_float(sf, s->samples_l, s->num_frames);
		memcpy(s->samples_r, s->samples_l, s->num_frames * sizeof(float));
		s->is_stereo = 0;
	} else {
		float *tmp = calloc(s->num_frames * info.channels, sizeof(float));
		sf_readf_float(sf, tmp, s->num_frames);
		for (int i = 0; i < s->num_frames; i++) {
			s->samples_l[i] = tmp[i * info.channels + 0];
			s->samples_r[i] = (info.channels >= 2)
				? tmp[i * info.channels + 1]
				: tmp[i * info.channels + 0];
		}
		free(tmp);
		s->is_stereo = 1;
	}
	sf_close(sf);

	if (info.samplerate != sr)
		fprintf(stderr, "Warning: %s is %dHz (expected %d)\n",
		        s->filename, info.samplerate, sr);
	return 0;
}

/* ---- frequency-domain convolution worker ---- */

static void *stem_worker(void *arg)
{
	Stem *s = (Stem *)arg;
	/*
	 * overlap-add convolution:
	 *   - take blocks of `block_size` input samples (no window)
	 *   - zero-pad to `fft_size` (block_size + ir_length - 1, rounded up)
	 *   - FFT, multiply by HRTF, IFFT
	 *   - overlap-add with stride = block_size
	 *
	 * fft_size=1024 accommodates block_size=512 + ir_length=512
	 */
	int N = HRTF_FFT_SIZE;    /* 1024 */
	int block_size = 512;     /* input samples per block */
	int n_bins = N / 2 + 1;
	int frames = s->num_frames;
	int out_len = frames + N; /* extra for convolution tail */

	s->out_frames = frames;
	s->out_l = calloc(out_len, sizeof(float));
	s->out_r = calloc(out_len, sizeof(float));

	float *block = calloc(N, sizeof(float));
	cpx *spec = calloc(n_bins, sizeof(cpx));
	cpx *conv_l = calloc(n_bins, sizeof(cpx));
	cpx *conv_r = calloc(n_bins, sizeof(cpx));
	float *result_l = calloc(N, sizeof(float));
	float *result_r = calloc(N, sizeof(float));

	/* stereo spread: offset L and R channels by ~15 degrees */
	float spread = 15.0f;
	cpx hrtf_ll[HRTF_N_BINS], hrtf_lr[HRTF_N_BINS];
	cpx hrtf_rl[HRTF_N_BINS], hrtf_rr[HRTF_N_BINS];

	if (s->is_stereo) {
		int idx_l = find_nearest_hrtf(s->az_deg - spread, s->el_deg);
		int idx_r = find_nearest_hrtf(s->az_deg + spread, s->el_deg);
		get_hrtf(idx_l, hrtf_ll, hrtf_lr);
		get_hrtf(idx_r, hrtf_rl, hrtf_rr);
	} else {
		int idx = find_nearest_hrtf(s->az_deg, s->el_deg);
		get_hrtf(idx, hrtf_ll, hrtf_lr);
		memcpy(hrtf_rl, hrtf_ll, sizeof(hrtf_rl));
		memcpy(hrtf_rr, hrtf_lr, sizeof(hrtf_rr));
	}

	cpx *spec2 = calloc(n_bins, sizeof(cpx));
	float *block2 = calloc(N, sizeof(float));

	for (int pos = 0; pos < frames; pos += block_size) {
		int copy_len = (pos + block_size <= frames) ? block_size : frames - pos;

		/* load L and R blocks, zero-padded */
		memset(block, 0, N * sizeof(float));
		memset(block2, 0, N * sizeof(float));
		for (int i = 0; i < copy_len; i++) {
			block[i]  = s->samples_l[pos + i] * s->stem_gain;
			block2[i] = s->samples_r[pos + i] * s->stem_gain;
		}

		/* FFT both channels */
		rfft(block, spec, N);
		rfft(block2, spec2, N);

		/* convolve L channel with its HRTF, R channel with its HRTF, sum into ears */
		for (int i = 0; i < n_bins; i++) {
			/* L input -> left ear + R input -> left ear */
			conv_l[i] = (cpx){
				(spec[i].re * hrtf_ll[i].re - spec[i].im * hrtf_ll[i].im)
				+ (spec2[i].re * hrtf_rl[i].re - spec2[i].im * hrtf_rl[i].im),
				(spec[i].re * hrtf_ll[i].im + spec[i].im * hrtf_ll[i].re)
				+ (spec2[i].re * hrtf_rl[i].im + spec2[i].im * hrtf_rl[i].re)
			};
			/* L input -> right ear + R input -> right ear */
			conv_r[i] = (cpx){
				(spec[i].re * hrtf_lr[i].re - spec[i].im * hrtf_lr[i].im)
				+ (spec2[i].re * hrtf_rr[i].re - spec2[i].im * hrtf_rr[i].im),
				(spec[i].re * hrtf_lr[i].im + spec[i].im * hrtf_lr[i].re)
				+ (spec2[i].re * hrtf_rr[i].im + spec2[i].im * hrtf_rr[i].re)
			};
		}

		irfft(conv_l, result_l, N);
		irfft(conv_r, result_r, N);

		for (int i = 0; i < N && pos + i < out_len; i++) {
			s->out_l[pos + i] += result_l[i];
			s->out_r[pos + i] += result_r[i];
		}
	}

	free(spec2); free(block2);

	free(block); free(spec);
	free(conv_l); free(conv_r);
	free(result_l); free(result_r);
	return NULL;
}

/* ---- arg parsing (same interface as spatialize.c) ---- */

static int parse_stem_arg(const char *arg, char **file, float *az, float *el, float *gain_db)
{
	*gain_db = 0.0f;
	const char *colons[4];
	int ncolons = 0;
	for (const char *p = arg + strlen(arg) - 1; p > arg && ncolons < 4; p--)
		if (*p == ':') colons[ncolons++] = p;

	if (ncolons < 2) return -1;

	const char *p_el, *p_az, *p_gain = NULL;
	if (ncolons >= 3) {
		p_gain = colons[0]; p_el = colons[1]; p_az = colons[2];
		char *end;
		strtod(p_az + 1, &end);
		if (end != p_el) {
			p_gain = NULL; p_el = colons[0]; p_az = colons[1];
		}
	} else {
		p_el = colons[0]; p_az = colons[1];
	}

	int flen = (int)(p_az - arg);
	*file = malloc(flen + 1);
	memcpy(*file, arg, flen);
	(*file)[flen] = '\0';

	*az = (float)atof(p_az + 1);
	*el = (float)atof(p_el + 1);
	if (p_gain) *gain_db = (float)atof(p_gain + 1);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "Usage: %s -o output.wav [-r wet] stem1.wav:az:el[:gain] ...\n", argv[0]);
		return 1;
	}

	const char *outfile = NULL;
	float reverb_wet = 0.12f;
	int stem_start = 1;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
			outfile = argv[++i]; stem_start = i + 1;
		} else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
			reverb_wet = (float)atof(argv[++i]); stem_start = i + 1;
		}
	}
	if (!outfile) { fprintf(stderr, "Missing -o\n"); return 1; }

	int nstem = argc - stem_start;
	if (nstem < 1) { fprintf(stderr, "No stems\n"); return 1; }

	Stem *stems = calloc(nstem, sizeof(Stem));
	for (int i = 0; i < nstem; i++) {
		float az, el, gain_db;
		char *file;
		if (parse_stem_arg(argv[stem_start + i], &file, &az, &el, &gain_db) < 0) {
			fprintf(stderr, "Bad arg: %s\n", argv[stem_start + i]);
			return 1;
		}
		stems[i].filename = file;
		stems[i].az_deg = az;
		stems[i].el_deg = el;
		stems[i].stem_gain = powf(10.0f, gain_db / 20.0f);
		printf("  stem %d: %s at az=%.0f° el=%.0f°\n", i, file, az, el);
		if (gain_db != 0.0f)
			printf("           gain=%.1fdB\n", gain_db);
	}

	int sample_rate = 0;
	for (int i = 0; i < nstem; i++) {
		SF_INFO info = {0};
		SNDFILE *sf = sf_open(stems[i].filename, SFM_READ, &info);
		if (sf) { if (!sample_rate) sample_rate = info.samplerate; sf_close(sf); }
	}
	if (!sample_rate) sample_rate = 44100;

	int max_frames = 0;
	for (int i = 0; i < nstem; i++) {
		if (load_stem(&stems[i], sample_rate) < 0) return 1;
		if (stems[i].num_frames > max_frames)
			max_frames = stems[i].num_frames;
	}

	/* look up HRTFs for each stem position */
	for (int i = 0; i < nstem; i++) {
		int idx = find_nearest_hrtf(stems[i].az_deg, stems[i].el_deg);
		get_hrtf(idx, stems[i].hrtf_l, stems[i].hrtf_r);
	}

	/* process stems in parallel */
	pthread_t *threads = calloc(nstem, sizeof(pthread_t));
	for (int i = 0; i < nstem; i++)
		pthread_create(&threads[i], NULL, stem_worker, &stems[i]);
	for (int i = 0; i < nstem; i++)
		pthread_join(threads[i], NULL);
	free(threads);

	/* merge + reverb */
	Reverb reverb;
	reverb_init(&reverb, (double)sample_rate, reverb_wet);

	float *out = calloc(max_frames * 2, sizeof(float));
	float peak = 0.0f;

	for (int n = 0; n < max_frames; n++) {
		float mix_l = 0.0f, mix_r = 0.0f;
		for (int i = 0; i < nstem; i++) {
			if (n < stems[i].out_frames) {
				mix_l += stems[i].out_l[n];
				mix_r += stems[i].out_r[n];
			}
		}

		float rev_l, rev_r;
		reverb_process(&reverb, mix_l, mix_r, &rev_l, &rev_r);
		out[n * 2 + 0] = rev_l;
		out[n * 2 + 1] = rev_r;

		float al = fabsf(rev_l), ar = fabsf(rev_r);
		if (al > peak) peak = al;
		if (ar > peak) peak = ar;
	}

	if (peak > 0.95f) {
		float scale = 0.95f / peak;
		for (int i = 0; i < max_frames * 2; i++)
			out[i] *= scale;
		printf("  normalized (peak was %.2f)\n", peak);
	}

	SF_INFO sfinfo = {
		.frames = max_frames, .samplerate = sample_rate,
		.channels = 2, .format = SF_FORMAT_WAV | SF_FORMAT_FLOAT,
	};
	SNDFILE *sf = sf_open(outfile, SFM_WRITE, &sfinfo);
	if (!sf) { fprintf(stderr, "Can't write %s\n", outfile); return 1; }
	sf_writef_float(sf, out, max_frames);
	sf_close(sf);

	printf("Wrote %d frames (%.1fs) to %s\n",
	       max_frames, (double)max_frames / sample_rate, outfile);

	for (int i = 0; i < nstem; i++) {
		free(stems[i].filename);
		free(stems[i].samples_l);
		free(stems[i].samples_r);
		free(stems[i].out_l);
		free(stems[i].out_r);
	}
	free(stems);
	free(out);
	reverb_free(&reverb);
	return 0;
}
