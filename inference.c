/*
 * inference.c — C inference engine for the stem separator model
 *
 * no Python, no PyTorch, no ONNX. just raw matrix math.
 * loads weights from a binary file exported by export_weights.py
 *
 * Compile: (via Makefile)
 * Usage: ./inference model.bin input.wav output_dir/
 */

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sndfile.h>
#include <sys/stat.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- basic tensor ops ---- */

/* y = x @ W^T + b  (x: [M,K], W: [N,K], b: [N], y: [M,N]) */
static void linear(float *y, const float *x, const float *w, const float *b,
                   int M, int K, int N)
{
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			float sum = b ? b[j] : 0.0f;
			for (int k = 0; k < K; k++)
				sum += x[i * K + k] * w[j * K + k];
			y[i * N + j] = sum;
		}
	}
}

/* GELU activation in-place */
static void gelu(float *x, int n)
{
	for (int i = 0; i < n; i++) {
		float v = x[i];
		x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
	}
}

/* sigmoid in-place */
static void sigmoid(float *x, int n)
{
	for (int i = 0; i < n; i++)
		x[i] = 1.0f / (1.0f + expf(-x[i]));
}

/* layer norm: normalize over last dim, apply scale and bias */
static void layer_norm(float *x, const float *gamma, const float *beta, int rows, int cols)
{
	for (int i = 0; i < rows; i++) {
		float *row = x + i * cols;
		float mean = 0, var = 0;
		for (int j = 0; j < cols; j++) mean += row[j];
		mean /= cols;
		for (int j = 0; j < cols; j++) { float d = row[j] - mean; var += d * d; }
		var /= cols;
		float inv = 1.0f / sqrtf(var + 1e-5f);
		for (int j = 0; j < cols; j++)
			row[j] = (row[j] - mean) * inv * gamma[j] + beta[j];
	}
}

/* softmax over last dim */
static void softmax(float *x, int rows, int cols)
{
	for (int i = 0; i < rows; i++) {
		float *row = x + i * cols;
		float max_val = row[0];
		for (int j = 1; j < cols; j++) if (row[j] > max_val) max_val = row[j];
		float sum = 0;
		for (int j = 0; j < cols; j++) { row[j] = expf(row[j] - max_val); sum += row[j]; }
		for (int j = 0; j < cols; j++) row[j] /= sum;
	}
}

/* multi-head attention: Q,K,V from in_proj, scaled dot product, out_proj
   x: [seq, emb], in_proj_w: [3*emb, emb], out_proj_w: [emb, emb] */
static void multihead_attention(float *out, float *x, int seq, int emb, int n_heads,
                                const float *in_proj_w, const float *in_proj_b,
                                const float *out_proj_w, const float *out_proj_b)
{
	int head_dim = emb / n_heads;
	float scale = 1.0f / sqrtf((float)head_dim);

	/* QKV projection: [seq, emb] @ [3*emb, emb]^T = [seq, 3*emb] */
	float *qkv = calloc(seq * 3 * emb, sizeof(float));
	linear(qkv, x, in_proj_w, in_proj_b, seq, emb, 3 * emb);

	float *Q = qkv;
	float *K = qkv + seq * emb;
	float *V = qkv + seq * 2 * emb;

	/* per-head attention */
	float *attn_out = calloc(seq * emb, sizeof(float));
	float *scores = calloc(seq * seq, sizeof(float));

	for (int h = 0; h < n_heads; h++) {
		int off = h * head_dim;

		/* scores = Q @ K^T * scale */
		for (int i = 0; i < seq; i++)
			for (int j = 0; j < seq; j++) {
				float s = 0;
				for (int k = 0; k < head_dim; k++)
					s += Q[i * emb + off + k] * K[j * emb + off + k];
				scores[i * seq + j] = s * scale;
			}

		softmax(scores, seq, seq);

		/* weighted sum of values */
		for (int i = 0; i < seq; i++)
			for (int k = 0; k < head_dim; k++) {
				float s = 0;
				for (int j = 0; j < seq; j++)
					s += scores[i * seq + j] * V[j * emb + off + k];
				attn_out[i * emb + off + k] = s;
			}
	}

	/* output projection */
	linear(out, attn_out, out_proj_w, out_proj_b, seq, emb, emb);

	free(qkv);
	free(attn_out);
	free(scores);
}

/* ---- FFT (same radix-2 as spatialize_fft.c) ---- */

typedef struct { float re, im; } cpx;

static void fft(cpx *buf, int n, int inverse)
{
	for (int i = 1, j = 0; i < n; i++) {
		int bit = n >> 1;
		for (; j & bit; bit >>= 1) j ^= bit;
		j ^= bit;
		if (i < j) { cpx tmp = buf[i]; buf[i] = buf[j]; buf[j] = tmp; }
	}
	for (int len = 2; len <= n; len <<= 1) {
		float ang = 2.0f * (float)M_PI / len * (inverse ? -1 : 1);
		cpx wlen = { cosf(ang), sinf(ang) };
		for (int i = 0; i < n; i += len) {
			cpx w = { 1.0f, 0.0f };
			for (int j = 0; j < len / 2; j++) {
				cpx u = buf[i + j];
				cpx v = {
					buf[i+j+len/2].re * w.re - buf[i+j+len/2].im * w.im,
					buf[i+j+len/2].re * w.im + buf[i+j+len/2].im * w.re
				};
				buf[i+j] = (cpx){ u.re + v.re, u.im + v.im };
				buf[i+j+len/2] = (cpx){ u.re - v.re, u.im - v.im };
				float wr = w.re * wlen.re - w.im * wlen.im;
				w.im = w.re * wlen.im + w.im * wlen.re;
				w.re = wr;
			}
		}
	}
	if (inverse) {
		float s = 1.0f / n;
		for (int i = 0; i < n; i++) { buf[i].re *= s; buf[i].im *= s; }
	}
}

/* STFT: input [samples], output [freq_bins, frames, 2] (real/imag interleaved per bin) */
static void stft(const float *input, int n_samples, int n_fft, int hop,
                const float *window, float *out_real, float *out_imag, int *out_frames)
{
	int n_bins = n_fft / 2 + 1;
	int frames = (n_samples - n_fft) / hop + 1;
	*out_frames = frames;

	cpx *buf = calloc(n_fft, sizeof(cpx));

	for (int t = 0; t < frames; t++) {
		int start = t * hop;
		for (int i = 0; i < n_fft; i++)
			buf[i] = (cpx){ input[start + i] * window[i], 0 };
		fft(buf, n_fft, 0);
		for (int f = 0; f < n_bins; f++) {
			out_real[f * frames + t] = buf[f].re;
			out_imag[f * frames + t] = buf[f].im;
		}
	}
	free(buf);
}

/* ISTFT: input [freq_bins, frames, 2], output [samples] */
static void istft(const float *in_real, const float *in_imag, int n_bins, int frames,
                 int n_fft, int hop, const float *window, float *output, int n_samples)
{
	memset(output, 0, n_samples * sizeof(float));
	float *win_sum = calloc(n_samples, sizeof(float));
	cpx *buf = calloc(n_fft, sizeof(cpx));

	for (int t = 0; t < frames; t++) {
		/* load spectrum */
		for (int f = 0; f < n_bins; f++)
			buf[f] = (cpx){ in_real[f * frames + t], in_imag[f * frames + t] };
		/* mirror conjugate */
		for (int f = 1; f < n_bins - 1; f++)
			buf[n_fft - f] = (cpx){ buf[f].re, -buf[f].im };

		fft(buf, n_fft, 1);

		int start = t * hop;
		for (int i = 0; i < n_fft && start + i < n_samples; i++) {
			output[start + i] += buf[i].re * window[i];
			win_sum[start + i] += window[i] * window[i];
		}
	}

	for (int i = 0; i < n_samples; i++)
		if (win_sum[i] > 1e-8f) output[i] /= win_sum[i];

	free(win_sum);
	free(buf);
}

/* ---- model weight loading ---- */

typedef struct {
	int sample_rate, n_fft, hop_length;
	int n_bands, emb_dim, n_heads, n_layers, n_sources;
	int band_starts[64], band_ends[64];

	/* all weights stored as flat arrays, indexed by layer */
	/* band_split: n_bands projections */
	float *bs_w[64], *bs_b[64];

	/* attention layers: n_layers of (band_norm, band_attn, band_ff, ..., time_*) */
	float *band_norm_g[16], *band_norm_b[16];
	float *band_attn_in_w[16], *band_attn_in_b[16];
	float *band_attn_out_w[16], *band_attn_out_b[16];
	float *band_ff_w1[16], *band_ff_b1[16];
	float *band_ff_w2[16], *band_ff_b2[16];
	float *band_ff_norm_g[16], *band_ff_norm_b[16];
	float *time_norm_g[16], *time_norm_b[16];
	float *time_attn_in_w[16], *time_attn_in_b[16];
	float *time_attn_out_w[16], *time_attn_out_b[16];
	float *time_ff_w1[16], *time_ff_b1[16];
	float *time_ff_w2[16], *time_ff_b2[16];
	float *time_ff_norm_g[16], *time_ff_norm_b[16];

	/* band_merge: n_bands projections */
	float *bm_w[64], *bm_b[64];
} Model;

static float *read_tensor(FILE *f, int *total_size)
{
	int name_len; fread(&name_len, 4, 1, f);
	char name[256]; fread(name, 1, name_len, f); name[name_len] = 0;

	int ndim; fread(&ndim, 4, 1, f);
	int shape[8], size = 1;
	for (int i = 0; i < ndim; i++) { fread(&shape[i], 4, 1, f); size *= shape[i]; }

	float *data = malloc(size * sizeof(float));
	fread(data, sizeof(float), size, f);

	if (total_size) *total_size = size;
	return data;
}

static Model *load_model(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) { fprintf(stderr, "can't open %s\n", path); return NULL; }

	char magic[4]; fread(magic, 1, 4, f);
	if (memcmp(magic, "STEM", 4) != 0) { fprintf(stderr, "bad magic\n"); fclose(f); return NULL; }

	Model *m = calloc(1, sizeof(Model));
	fread(&m->sample_rate, 4, 1, f);
	fread(&m->n_fft, 4, 1, f);
	fread(&m->hop_length, 4, 1, f);
	fread(&m->n_bands, 4, 1, f);
	fread(&m->emb_dim, 4, 1, f);
	fread(&m->n_heads, 4, 1, f);
	fread(&m->n_layers, 4, 1, f);
	fread(&m->n_sources, 4, 1, f);

	for (int i = 0; i < m->n_bands; i++) {
		fread(&m->band_starts[i], 4, 1, f);
		fread(&m->band_ends[i], 4, 1, f);
	}

	int n_params; fread(&n_params, 4, 1, f);

	/* read parameters in order — they come in a fixed sequence from PyTorch */
	for (int i = 0; i < m->n_bands; i++) {
		m->bs_w[i] = read_tensor(f, NULL);
		m->bs_b[i] = read_tensor(f, NULL);
	}

	for (int l = 0; l < m->n_layers; l++) {
		m->band_norm_g[l]    = read_tensor(f, NULL);
		m->band_norm_b[l]    = read_tensor(f, NULL);
		m->band_attn_in_w[l] = read_tensor(f, NULL);
		m->band_attn_in_b[l] = read_tensor(f, NULL);
		m->band_attn_out_w[l]= read_tensor(f, NULL);
		m->band_attn_out_b[l]= read_tensor(f, NULL);
		m->band_ff_w1[l]     = read_tensor(f, NULL);
		m->band_ff_b1[l]     = read_tensor(f, NULL);
		/* skip dropout (no weights) */
		m->band_ff_w2[l]     = read_tensor(f, NULL);
		m->band_ff_b2[l]     = read_tensor(f, NULL);
		/* skip dropout */
		m->band_ff_norm_g[l] = read_tensor(f, NULL);
		m->band_ff_norm_b[l] = read_tensor(f, NULL);
		m->time_norm_g[l]    = read_tensor(f, NULL);
		m->time_norm_b[l]    = read_tensor(f, NULL);
		m->time_attn_in_w[l] = read_tensor(f, NULL);
		m->time_attn_in_b[l] = read_tensor(f, NULL);
		m->time_attn_out_w[l]= read_tensor(f, NULL);
		m->time_attn_out_b[l]= read_tensor(f, NULL);
		m->time_ff_w1[l]     = read_tensor(f, NULL);
		m->time_ff_b1[l]     = read_tensor(f, NULL);
		m->time_ff_w2[l]     = read_tensor(f, NULL);
		m->time_ff_b2[l]     = read_tensor(f, NULL);
		m->time_ff_norm_g[l] = read_tensor(f, NULL);
		m->time_ff_norm_b[l] = read_tensor(f, NULL);
	}

	for (int i = 0; i < m->n_bands; i++) {
		m->bm_w[i] = read_tensor(f, NULL);
		m->bm_b[i] = read_tensor(f, NULL);
	}

	fclose(f);
	return m;
}

/* ---- forward pass ---- */

/* process one chunk of stereo audio through the model
   input: [2, samples], output: [n_sources, 2, samples] */
static void forward_chunk(Model *m, const float *input_l, const float *input_r,
                          int n_samples, float **out_sources)
{
	int n_fft = m->n_fft;
	int hop = m->hop_length;
	int n_bins = n_fft / 2 + 1;
	int emb = m->emb_dim;
	int n_bands = m->n_bands;
	int n_sources = m->n_sources;

	/* hann window */
	float *window = malloc(n_fft * sizeof(float));
	for (int i = 0; i < n_fft; i++)
		window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n_fft));

	/* STFT both channels */
	int T;
	float *spec_lr = calloc(2 * n_bins * 1, sizeof(float)); /* placeholder */
	float *ch_real[2], *ch_imag[2];
	for (int c = 0; c < 2; c++) {
		const float *ch_in = (c == 0) ? input_l : input_r;
		ch_real[c] = calloc(n_bins * ((n_samples - n_fft) / hop + 1), sizeof(float));
		ch_imag[c] = calloc(n_bins * ((n_samples - n_fft) / hop + 1), sizeof(float));
		stft(ch_in, n_samples, n_fft, hop, window, ch_real[c], ch_imag[c], &T);
	}
	free(spec_lr);

	/* band_split: for each band, gather [ch0_real, ch0_imag, ch1_real, ch1_imag]
	   input to each band projection: (T, band_size * 4) */
	float *bands_out = calloc(n_bands * T * emb, sizeof(float));

	for (int b = 0; b < n_bands; b++) {
		int start = m->band_starts[b], end = m->band_ends[b];
		int bsz = end - start;
		int in_dim = bsz * 4; /* 2 channels * 2 (real/imag) */

		float *band_in = calloc(T * in_dim, sizeof(float));
		for (int t = 0; t < T; t++) {
			float *row = band_in + t * in_dim;
			int off = 0;
			/* ch0 real */
			for (int f = start; f < end; f++) row[off++] = ch_real[0][f * T + t];
			/* ch0 imag */
			for (int f = start; f < end; f++) row[off++] = ch_imag[0][f * T + t];
			/* ch1 real */
			for (int f = start; f < end; f++) row[off++] = ch_real[1][f * T + t];
			/* ch1 imag */
			for (int f = start; f < end; f++) row[off++] = ch_imag[1][f * T + t];
		}

		/* linear + gelu */
		float *band_out = bands_out + b * T * emb;
		linear(band_out, band_in, m->bs_w[b], m->bs_b[b], T, in_dim, emb);
		gelu(band_out, T * emb);

		free(band_in);
	}

	/* interleaved attention: bands_out is [n_bands, T, emb] (contiguous per band) */
	float *x = bands_out; /* [n_bands * T * emb] */

	for (int l = 0; l < m->n_layers; l++) {
		/* --- band attention (at each time step, attend across bands) --- */
		/* reshape to [T, n_bands, emb], process each time step */
		float *resid = calloc(n_bands * T * emb, sizeof(float));
		float *normed = calloc(n_bands * emb, sizeof(float));
		float *attn_out = calloc(n_bands * emb, sizeof(float));

		for (int t = 0; t < T; t++) {
			/* gather bands for this time step: x[b][t] for all b */
			for (int b = 0; b < n_bands; b++)
				memcpy(normed + b * emb, x + b * T * emb + t * emb, emb * sizeof(float));

			layer_norm(normed, m->band_norm_g[l], m->band_norm_b[l], n_bands, emb);
			multihead_attention(attn_out, normed, n_bands, emb, m->n_heads,
				m->band_attn_in_w[l], m->band_attn_in_b[l],
				m->band_attn_out_w[l], m->band_attn_out_b[l]);

			/* residual: x += attn(norm(x)) */
			for (int b = 0; b < n_bands; b++)
				for (int e = 0; e < emb; e++) {
					int idx = b * T * emb + t * emb + e;
					resid[idx] = x[idx] + attn_out[b * emb + e];
				}
		}
		memcpy(x, resid, n_bands * T * emb * sizeof(float));

		/* band feedforward with residual */
		float *ff_in = calloc(n_bands * T * emb, sizeof(float));
		memcpy(ff_in, x, n_bands * T * emb * sizeof(float));
		layer_norm(ff_in, m->band_ff_norm_g[l], m->band_ff_norm_b[l], n_bands * T, emb);

		float *ff_mid = calloc(n_bands * T * emb * 2, sizeof(float));
		linear(ff_mid, ff_in, m->band_ff_w1[l], m->band_ff_b1[l], n_bands * T, emb, emb * 2);
		gelu(ff_mid, n_bands * T * emb * 2);
		float *ff_out = calloc(n_bands * T * emb, sizeof(float));
		linear(ff_out, ff_mid, m->band_ff_w2[l], m->band_ff_b2[l], n_bands * T, emb * 2, emb);

		for (int i = 0; i < n_bands * T * emb; i++) x[i] += ff_out[i];

		free(resid); free(normed); free(attn_out);
		free(ff_in); free(ff_mid); free(ff_out);

		/* --- time attention (for each band, attend across time) --- */
		resid = calloc(n_bands * T * emb, sizeof(float));
		normed = calloc(T * emb, sizeof(float));
		attn_out = calloc(T * emb, sizeof(float));

		for (int b = 0; b < n_bands; b++) {
			float *band_data = x + b * T * emb;
			memcpy(normed, band_data, T * emb * sizeof(float));
			layer_norm(normed, m->time_norm_g[l], m->time_norm_b[l], T, emb);
			multihead_attention(attn_out, normed, T, emb, m->n_heads,
				m->time_attn_in_w[l], m->time_attn_in_b[l],
				m->time_attn_out_w[l], m->time_attn_out_b[l]);

			for (int i = 0; i < T * emb; i++)
				resid[b * T * emb + i] = band_data[i] + attn_out[i];
		}
		memcpy(x, resid, n_bands * T * emb * sizeof(float));

		/* time feedforward with residual */
		ff_in = calloc(n_bands * T * emb, sizeof(float));
		memcpy(ff_in, x, n_bands * T * emb * sizeof(float));
		layer_norm(ff_in, m->time_ff_norm_g[l], m->time_ff_norm_b[l], n_bands * T, emb);

		ff_mid = calloc(n_bands * T * emb * 2, sizeof(float));
		linear(ff_mid, ff_in, m->time_ff_w1[l], m->time_ff_b1[l], n_bands * T, emb, emb * 2);
		gelu(ff_mid, n_bands * T * emb * 2);
		ff_out = calloc(n_bands * T * emb, sizeof(float));
		linear(ff_out, ff_mid, m->time_ff_w2[l], m->time_ff_b2[l], n_bands * T, emb * 2, emb);

		for (int i = 0; i < n_bands * T * emb; i++) x[i] += ff_out[i];

		free(resid); free(normed); free(attn_out);
		free(ff_in); free(ff_mid); free(ff_out);
	}

	/* band_merge: project back to spectrogram masks */
	/* for each band, project [T, emb] -> [T, band_size * n_channels * 2 * n_sources] */
	/* then sigmoid */
	/* masks layout: [n_sources][2 channels][2 ri][freq][T] */
	float *masks_real = calloc(n_sources * 2 * n_bins * T, sizeof(float));
	float *masks_imag = calloc(n_sources * 2 * n_bins * T, sizeof(float));

	for (int b = 0; b < n_bands; b++) {
		int start = m->band_starts[b], end = m->band_ends[b];
		int bsz = end - start;
		int out_dim = bsz * 2 * 2 * n_sources; /* band_size * channels * ri * sources */

		float *proj_out = calloc(T * out_dim, sizeof(float));
		linear(proj_out, x + b * T * emb, m->bm_w[b], m->bm_b[b], T, emb, out_dim);
		sigmoid(proj_out, T * out_dim);

		/* scatter into masks */
		for (int t = 0; t < T; t++) {
			float *row = proj_out + t * out_dim;
			int off = 0;
			for (int s = 0; s < n_sources; s++)
				for (int c = 0; c < 2; c++) {
					for (int f = 0; f < bsz; f++) {
						int freq = start + f;
						int mi = (s * 2 + c) * n_bins * T + freq * T + t;
						masks_real[mi] = row[off++];
					}
					for (int f = 0; f < bsz; f++) {
						int freq = start + f;
						int mi = (s * 2 + c) * n_bins * T + freq * T + t;
						masks_imag[mi] = row[off++];
					}
				}
		}
		free(proj_out);
	}

	/* apply masks: out = orig * mask (complex multiply using real arithmetic) */
	for (int s = 0; s < n_sources; s++) {
		for (int c = 0; c < 2; c++) {
			int si = (s * 2 + c) * n_bins * T;
			float *src_real = calloc(n_bins * T, sizeof(float));
			float *src_imag = calloc(n_bins * T, sizeof(float));

			for (int i = 0; i < n_bins * T; i++) {
				float or_ = (c == 0) ? ch_real[0][i] : ch_real[1][i];
				float oi  = (c == 0) ? ch_imag[0][i] : ch_imag[1][i];
				float mr  = masks_real[si + i];
				float mi  = masks_imag[si + i];
				src_real[i] = or_ * mr - oi * mi;
				src_imag[i] = or_ * mi + oi * mr;
			}

			/* ISTFT */
			istft(src_real, src_imag, n_bins, T, n_fft, hop, window,
			      out_sources[s * 2 + c], n_samples);

			free(src_real);
			free(src_imag);
		}
	}

	free(masks_real);
	free(masks_imag);
	free(ch_real[0]); free(ch_real[1]);
	free(ch_imag[0]); free(ch_imag[1]);
	free(bands_out);
	free(window);
}

/* ---- main ---- */

int main(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "usage: %s model.bin input.wav output_dir/\n", argv[0]);
		return 1;
	}

	const char *model_path = argv[1];
	const char *input_path = argv[2];
	const char *output_dir = argv[3];

	mkdir(output_dir, 0755);

	printf("loading model...\n");
	Model *m = load_model(model_path);
	if (!m) return 1;
	printf("  %d bands, %d layers, %d emb, %d sources\n",
	       m->n_bands, m->n_layers, m->emb_dim, m->n_sources);

	/* load audio */
	SF_INFO info = {0};
	SNDFILE *sf = sf_open(input_path, SFM_READ, &info);
	if (!sf) { fprintf(stderr, "can't open %s\n", input_path); return 1; }

	int n_samples = (int)info.frames;
	float *audio_l = calloc(n_samples, sizeof(float));
	float *audio_r = calloc(n_samples, sizeof(float));

	if (info.channels == 1) {
		sf_readf_float(sf, audio_l, n_samples);
		memcpy(audio_r, audio_l, n_samples * sizeof(float));
	} else {
		float *tmp = calloc(n_samples * info.channels, sizeof(float));
		sf_readf_float(sf, tmp, n_samples);
		for (int i = 0; i < n_samples; i++) {
			audio_l[i] = tmp[i * info.channels];
			audio_r[i] = tmp[i * info.channels + 1];
		}
		free(tmp);
	}
	sf_close(sf);

	float duration = (float)n_samples / m->sample_rate;
	printf("separating %.1fs of audio...\n", duration);

	/* allocate output: n_sources * 2 channels */
	int n_out = m->n_sources * 2;
	float **out = calloc(n_out, sizeof(float *));
	for (int i = 0; i < n_out; i++)
		out[i] = calloc(n_samples, sizeof(float));

	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);

	forward_chunk(m, audio_l, audio_r, n_samples, out);

	clock_gettime(CLOCK_MONOTONIC, &t1);
	float elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9f;
	printf("done in %.1fs (%.1fx realtime)\n", elapsed, duration / elapsed);

	/* save stems */
	const char *source_names[] = { "drums", "bass", "other", "vocals" };
	for (int s = 0; s < m->n_sources; s++) {
		char path[512];
		snprintf(path, sizeof(path), "%s/%s.wav", output_dir, source_names[s]);

		float *stereo = calloc(n_samples * 2, sizeof(float));
		for (int i = 0; i < n_samples; i++) {
			stereo[i * 2 + 0] = out[s * 2 + 0][i];
			stereo[i * 2 + 1] = out[s * 2 + 1][i];
		}

		SF_INFO out_info = {
			.frames = n_samples, .samplerate = m->sample_rate,
			.channels = 2, .format = SF_FORMAT_WAV | SF_FORMAT_FLOAT,
		};
		SNDFILE *out_sf = sf_open(path, SFM_WRITE, &out_info);
		if (out_sf) {
			sf_writef_float(out_sf, stereo, n_samples);
			sf_close(out_sf);
			printf("  saved %s\n", source_names[s]);
		}
		free(stereo);
	}

	/* cleanup */
	free(audio_l); free(audio_r);
	for (int i = 0; i < n_out; i++) free(out[i]);
	free(out);
	/* model cleanup omitted for brevity */

	return 0;
}
