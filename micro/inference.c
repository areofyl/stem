/*
 * micro/inference.c — C inference for the 4 micro separator models
 *
 * runs 4 tiny GRU models in parallel (one per stem) using pthreads.
 * each model: STFT -> encoder -> GRU -> decoder -> mask -> ISTFT
 * no attention, no transformers, just linear layers and GRU.
 *
 * usage: ./inference vocals.bin drums.bin bass.bin other.bin input.wav output_dir/
 */

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sndfile.h>
#include <sys/stat.h>
#include <time.h>
#include <cblas.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- basic ops ---- */

static void linear(float *y, const float *x, const float *w, const float *b,
                   int M, int K, int N)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
	            M, N, K, 1.0f, x, K, w, K, 0.0f, y, N);
	if (b)
		for (int i = 0; i < M; i++)
			cblas_saxpy(N, 1.0f, b, 1, y + i * N, 1);
}

static void relu(float *x, int n)
{
	for (int i = 0; i < n; i++)
		if (x[i] < 0) x[i] = 0;
}

static void sigmoid_inplace(float *x, int n)
{
	for (int i = 0; i < n; i++)
		x[i] = 1.0f / (1.0f + expf(-x[i]));
}

/* GRU cell: one time step
   x_t: [hidden], h: [hidden] (modified in place)
   weight_ih: [3*hidden, input], weight_hh: [3*hidden, hidden]
   bias_ih: [3*hidden], bias_hh: [3*hidden] */
static void gru_step(float *h, const float *x_t, int input_dim, int hidden,
                     const float *w_ih, const float *w_hh,
                     const float *b_ih, const float *b_hh)
{
	int h3 = 3 * hidden;
	float *gates_i = calloc(h3, sizeof(float));
	float *gates_h = calloc(h3, sizeof(float));

	/* gates_i = x_t @ w_ih^T + b_ih */
	cblas_sgemv(CblasRowMajor, CblasNoTrans, h3, input_dim, 1.0f,
	            w_ih, input_dim, x_t, 1, 0.0f, gates_i, 1);
	cblas_saxpy(h3, 1.0f, b_ih, 1, gates_i, 1);

	/* gates_h = h @ w_hh^T + b_hh */
	cblas_sgemv(CblasRowMajor, CblasNoTrans, h3, hidden, 1.0f,
	            w_hh, hidden, h, 1, 0.0f, gates_h, 1);
	cblas_saxpy(h3, 1.0f, b_hh, 1, gates_h, 1);

	/* r = sigmoid(gates_i[0:h] + gates_h[0:h])
	   z = sigmoid(gates_i[h:2h] + gates_h[h:2h])
	   n = tanh(gates_i[2h:3h] + r * gates_h[2h:3h])
	   h = (1-z) * n + z * h */
	for (int i = 0; i < hidden; i++) {
		float r = 1.0f / (1.0f + expf(-(gates_i[i] + gates_h[i])));
		float z = 1.0f / (1.0f + expf(-(gates_i[hidden + i] + gates_h[hidden + i])));
		float n = tanhf(gates_i[2*hidden + i] + r * gates_h[2*hidden + i]);
		h[i] = (1.0f - z) * n + z * h[i];
	}

	free(gates_i);
	free(gates_h);
}

/* ---- FFT ---- */

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

/* ---- model ---- */

typedef struct {
	int n_fft, hop, hidden, n_gru_layers;
	int n_bins; /* n_fft/2 + 1 */

	/* encoder: 2 linear layers */
	float *enc_w1, *enc_b1; /* [hidden, n_bins] */
	float *enc_w2, *enc_b2; /* [hidden, hidden] */

	/* GRU: 2 layers */
	float *gru_wih[2], *gru_whh[2];
	float *gru_bih[2], *gru_bhh[2];

	/* decoder: 2 linear layers */
	float *dec_w1, *dec_b1; /* [hidden, hidden] */
	float *dec_w2, *dec_b2; /* [n_bins, hidden] */
} MicroModel;

static float *read_tensor(FILE *f)
{
	int name_len; fread(&name_len, 4, 1, f);
	char name[256]; fread(name, 1, name_len, f);
	int ndim; fread(&ndim, 4, 1, f);
	int size = 1;
	for (int i = 0; i < ndim; i++) { int s; fread(&s, 4, 1, f); size *= s; }
	float *data = malloc(size * sizeof(float));
	fread(data, sizeof(float), size, f);
	return data;
}

static MicroModel *load_model(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;

	char magic[4]; fread(magic, 1, 4, f);
	if (memcmp(magic, "MICR", 4) != 0) { fclose(f); return NULL; }

	MicroModel *m = calloc(1, sizeof(MicroModel));
	fread(&m->n_fft, 4, 1, f);
	fread(&m->hop, 4, 1, f);
	fread(&m->hidden, 4, 1, f);
	fread(&m->n_gru_layers, 4, 1, f);
	m->n_bins = m->n_fft / 2 + 1;

	int n_tensors; fread(&n_tensors, 4, 1, f);

	m->enc_w1 = read_tensor(f); m->enc_b1 = read_tensor(f);
	m->enc_w2 = read_tensor(f); m->enc_b2 = read_tensor(f);
	for (int l = 0; l < 2; l++) {
		m->gru_wih[l] = read_tensor(f); m->gru_whh[l] = read_tensor(f);
		m->gru_bih[l] = read_tensor(f); m->gru_bhh[l] = read_tensor(f);
	}
	m->dec_w1 = read_tensor(f); m->dec_b1 = read_tensor(f);
	m->dec_w2 = read_tensor(f); m->dec_b2 = read_tensor(f);

	fclose(f);
	return m;
}

/* ---- separation job (one per thread) ---- */

typedef struct {
	MicroModel *model;
	const float *input;  /* mono audio */
	int n_samples;
	int sample_rate;
	float *output;       /* result */
	const char *name;
	float elapsed;
} SepJob;

static void *separate_thread(void *arg)
{
	SepJob *job = arg;
	MicroModel *m = job->model;
	int n = job->n_samples;
	int n_fft = m->n_fft;
	int hop = m->hop;
	int n_bins = m->n_bins;
	int hidden = m->hidden;

	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);

	/* hann window */
	float *win = malloc(n_fft * sizeof(float));
	for (int i = 0; i < n_fft; i++)
		win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n_fft));

	/* STFT */
	int T = (n - n_fft) / hop + 1;
	float *mag = calloc(n_bins * T, sizeof(float));
	float *phase_re = calloc(n_bins * T, sizeof(float));
	float *phase_im = calloc(n_bins * T, sizeof(float));
	cpx *fft_buf = calloc(n_fft, sizeof(cpx));

	for (int t = 0; t < T; t++) {
		for (int i = 0; i < n_fft; i++)
			fft_buf[i] = (cpx){ job->input[t * hop + i] * win[i], 0 };
		fft(fft_buf, n_fft, 0);
		for (int f = 0; f < n_bins; f++) {
			float re = fft_buf[f].re, im = fft_buf[f].im;
			mag[f * T + t] = sqrtf(re * re + im * im);
			phase_re[f * T + t] = re;
			phase_im[f * T + t] = im;
		}
	}

	/* process frame-by-frame through encoder -> GRU -> decoder */
	float *h0 = calloc(hidden, sizeof(float)); /* GRU hidden state layer 0 */
	float *h1 = calloc(hidden, sizeof(float)); /* GRU hidden state layer 1 */
	float *enc_out = calloc(hidden, sizeof(float));
	float *dec_out = calloc(n_bins, sizeof(float));
	float *frame_in = calloc(n_bins, sizeof(float));
	float *mask = calloc(n_bins * T, sizeof(float));

	for (int t = 0; t < T; t++) {
		/* gather magnitude frame */
		for (int f = 0; f < n_bins; f++)
			frame_in[f] = mag[f * T + t];

		/* encoder: linear -> relu -> linear -> relu */
		linear(enc_out, frame_in, m->enc_w1, m->enc_b1, 1, n_bins, hidden);
		relu(enc_out, hidden);
		float *enc_tmp = calloc(hidden, sizeof(float));
		linear(enc_tmp, enc_out, m->enc_w2, m->enc_b2, 1, hidden, hidden);
		relu(enc_tmp, hidden);
		memcpy(enc_out, enc_tmp, hidden * sizeof(float));
		free(enc_tmp);

		/* GRU layer 0 */
		gru_step(h0, enc_out, hidden, hidden,
		         m->gru_wih[0], m->gru_whh[0], m->gru_bih[0], m->gru_bhh[0]);

		/* GRU layer 1 */
		gru_step(h1, h0, hidden, hidden,
		         m->gru_wih[1], m->gru_whh[1], m->gru_bih[1], m->gru_bhh[1]);

		/* decoder: linear -> relu -> linear -> sigmoid */
		float *dec_tmp = calloc(hidden, sizeof(float));
		linear(dec_tmp, h1, m->dec_w1, m->dec_b1, 1, hidden, hidden);
		relu(dec_tmp, hidden);
		linear(dec_out, dec_tmp, m->dec_w2, m->dec_b2, 1, hidden, n_bins);
		sigmoid_inplace(dec_out, n_bins);
		free(dec_tmp);

		/* store mask */
		for (int f = 0; f < n_bins; f++)
			mask[f * T + t] = dec_out[f];
	}

	/* apply mask and ISTFT */
	job->output = calloc(n, sizeof(float));
	float *win_sum = calloc(n, sizeof(float));

	for (int t = 0; t < T; t++) {
		/* apply mask to complex spectrum */
		for (int f = 0; f < n_bins; f++) {
			float m_val = mask[f * T + t];
			fft_buf[f] = (cpx){ phase_re[f * T + t] * m_val, phase_im[f * T + t] * m_val };
		}
		/* mirror conjugate */
		for (int f = 1; f < n_bins - 1; f++)
			fft_buf[n_fft - f] = (cpx){ fft_buf[f].re, -fft_buf[f].im };

		fft(fft_buf, n_fft, 1);

		int start = t * hop;
		for (int i = 0; i < n_fft && start + i < n; i++) {
			job->output[start + i] += fft_buf[i].re * win[i];
			win_sum[start + i] += win[i] * win[i];
		}
	}

	for (int i = 0; i < n; i++)
		if (win_sum[i] > 1e-8f) job->output[i] /= win_sum[i];

	free(win); free(mag); free(phase_re); free(phase_im);
	free(fft_buf); free(h0); free(h1);
	free(enc_out); free(dec_out); free(frame_in);
	free(mask); free(win_sum);

	clock_gettime(CLOCK_MONOTONIC, &t1);
	job->elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9f;
	return NULL;
}

/* ---- main ---- */

int main(int argc, char **argv)
{
	if (argc < 7) {
		fprintf(stderr, "usage: %s vocals.bin drums.bin bass.bin other.bin input.wav output_dir/\n", argv[0]);
		return 1;
	}

	const char *model_paths[] = { argv[1], argv[2], argv[3], argv[4] };
	const char *stem_names[] = { "vocals", "drums", "bass", "other" };
	const char *input_path = argv[5];
	const char *output_dir = argv[6];
	mkdir(output_dir, 0755);

	/* load all 4 models */
	MicroModel *models[4];
	for (int i = 0; i < 4; i++) {
		models[i] = load_model(model_paths[i]);
		if (!models[i]) { fprintf(stderr, "can't load %s\n", model_paths[i]); return 1; }
	}
	printf("loaded 4 micro models (%dK params each)\n", 297);

	/* load audio as mono */
	SF_INFO info = {0};
	SNDFILE *sf = sf_open(input_path, SFM_READ, &info);
	if (!sf) { fprintf(stderr, "can't open %s\n", input_path); return 1; }

	int n_samples = (int)info.frames;
	float *mono = calloc(n_samples, sizeof(float));

	if (info.channels == 1) {
		sf_readf_float(sf, mono, n_samples);
	} else {
		float *tmp = calloc(n_samples * info.channels, sizeof(float));
		sf_readf_float(sf, tmp, n_samples);
		for (int i = 0; i < n_samples; i++) {
			float sum = 0;
			for (int c = 0; c < info.channels; c++)
				sum += tmp[i * info.channels + c];
			mono[i] = sum / info.channels;
		}
		free(tmp);
	}
	sf_close(sf);

	float duration = (float)n_samples / info.samplerate;
	printf("separating %.1fs of audio...\n", duration);

	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);

	/* launch 4 threads */
	SepJob jobs[4];
	pthread_t threads[4];
	for (int i = 0; i < 4; i++) {
		jobs[i] = (SepJob){
			.model = models[i],
			.input = mono,
			.n_samples = n_samples,
			.sample_rate = info.samplerate,
			.name = stem_names[i],
		};
		pthread_create(&threads[i], NULL, separate_thread, &jobs[i]);
	}
	for (int i = 0; i < 4; i++)
		pthread_join(threads[i], NULL);

	clock_gettime(CLOCK_MONOTONIC, &t1);
	float total = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9f;
	printf("done in %.1fs (%.0fx realtime)\n", total, duration / total);

	/* save stems */
	for (int i = 0; i < 4; i++) {
		char path[512];
		snprintf(path, sizeof(path), "%s/%s.wav", output_dir, stem_names[i]);

		SF_INFO out_info = {
			.frames = n_samples, .samplerate = info.samplerate,
			.channels = 1, .format = SF_FORMAT_WAV | SF_FORMAT_FLOAT,
		};
		SNDFILE *out_sf = sf_open(path, SFM_WRITE, &out_info);
		if (out_sf) {
			sf_writef_float(out_sf, jobs[i].output, n_samples);
			sf_close(out_sf);
			printf("  %s (%.1fs)\n", stem_names[i], jobs[i].elapsed);
		}
		free(jobs[i].output);
	}

	free(mono);
	return 0;
}
