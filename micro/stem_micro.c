/*
 * stem_micro.c — complete pipeline in one binary
 *
 * reads any audio format (via libsndfile), separates with 4 micro models
 * in parallel, spatializes with HRTF, auto-glue, loudness match, writes output.
 *
 * no python, no ffmpeg, no subprocesses.
 *
 * usage: ./stem_micro -o out.wav [-t seconds] input.opus
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

/* ============ micro model inference (from micro/inference.c) ============ */

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
	for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}

static void sigmoid_inplace(float *x, int n)
{
	for (int i = 0; i < n; i++) x[i] = 1.0f / (1.0f + expf(-x[i]));
}

static void gru_step(float *h, const float *x_t, int input_dim, int hidden,
                     const float *w_ih, const float *w_hh,
                     const float *b_ih, const float *b_hh)
{
	int h3 = 3 * hidden;
	float *gates_i = calloc(h3, sizeof(float));
	float *gates_h = calloc(h3, sizeof(float));

	cblas_sgemv(CblasRowMajor, CblasNoTrans, h3, input_dim, 1.0f,
	            w_ih, input_dim, x_t, 1, 0.0f, gates_i, 1);
	cblas_saxpy(h3, 1.0f, b_ih, 1, gates_i, 1);
	cblas_sgemv(CblasRowMajor, CblasNoTrans, h3, hidden, 1.0f,
	            w_hh, hidden, h, 1, 0.0f, gates_h, 1);
	cblas_saxpy(h3, 1.0f, b_hh, 1, gates_h, 1);

	for (int i = 0; i < hidden; i++) {
		float r = 1.0f / (1.0f + expf(-(gates_i[i] + gates_h[i])));
		float z = 1.0f / (1.0f + expf(-(gates_i[hidden+i] + gates_h[hidden+i])));
		float n = tanhf(gates_i[2*hidden+i] + r * gates_h[2*hidden+i]);
		h[i] = (1.0f - z) * n + z * h[i];
	}
	free(gates_i); free(gates_h);
}

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
				cpx u = buf[i+j];
				cpx v = {
					buf[i+j+len/2].re*w.re - buf[i+j+len/2].im*w.im,
					buf[i+j+len/2].re*w.im + buf[i+j+len/2].im*w.re };
				buf[i+j] = (cpx){ u.re+v.re, u.im+v.im };
				buf[i+j+len/2] = (cpx){ u.re-v.re, u.im-v.im };
				float wr = w.re*wlen.re - w.im*wlen.im;
				w.im = w.re*wlen.im + w.im*wlen.re;
				w.re = wr;
			}
		}
	}
	if (inverse) {
		float s = 1.0f / n;
		for (int i = 0; i < n; i++) { buf[i].re *= s; buf[i].im *= s; }
	}
}

typedef struct {
	int n_fft, hop, hidden, n_gru_layers, n_bins;
	float *enc_w1, *enc_b1, *enc_w2, *enc_b2;
	float *gru_wih[2], *gru_whh[2], *gru_bih[2], *gru_bhh[2];
	float *dec_w1, *dec_b1, *dec_w2, *dec_b2;
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

static MicroModel *load_micro(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	char magic[4]; fread(magic, 1, 4, f);
	if (memcmp(magic, "MICR", 4)) { fclose(f); return NULL; }
	MicroModel *m = calloc(1, sizeof(MicroModel));
	fread(&m->n_fft, 4, 1, f); fread(&m->hop, 4, 1, f);
	fread(&m->hidden, 4, 1, f); fread(&m->n_gru_layers, 4, 1, f);
	m->n_bins = m->n_fft / 2 + 1;
	int nt; fread(&nt, 4, 1, f);
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

typedef struct {
	MicroModel *model;
	const float *input;
	int n_samples;
	float *output;
} SepJob;

static void *separate_thread(void *arg)
{
	SepJob *job = arg;
	MicroModel *m = job->model;
	int n = job->n_samples;
	int n_fft = m->n_fft, hop = m->hop, n_bins = m->n_bins, hidden = m->hidden;

	float *win = malloc(n_fft * sizeof(float));
	for (int i = 0; i < n_fft; i++)
		win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n_fft));

	int T = (n - n_fft) / hop + 1;
	float *mag = calloc(n_bins * T, sizeof(float));
	float *ph_re = calloc(n_bins * T, sizeof(float));
	float *ph_im = calloc(n_bins * T, sizeof(float));
	cpx *fb = calloc(n_fft, sizeof(cpx));

	for (int t = 0; t < T; t++) {
		for (int i = 0; i < n_fft; i++)
			fb[i] = (cpx){ job->input[t*hop+i] * win[i], 0 };
		fft(fb, n_fft, 0);
		for (int f = 0; f < n_bins; f++) {
			ph_re[f*T+t] = fb[f].re;
			ph_im[f*T+t] = fb[f].im;
			mag[f*T+t] = sqrtf(fb[f].re*fb[f].re + fb[f].im*fb[f].im);
		}
	}

	float *h0 = calloc(hidden, sizeof(float));
	float *h1 = calloc(hidden, sizeof(float));
	float *enc = calloc(hidden, sizeof(float));
	float *enc2 = calloc(hidden, sizeof(float));
	float *dec = calloc(n_bins, sizeof(float));
	float *dec2 = calloc(hidden, sizeof(float));
	float *fin = calloc(n_bins, sizeof(float));
	float *mask = calloc(n_bins * T, sizeof(float));

	for (int t = 0; t < T; t++) {
		for (int f = 0; f < n_bins; f++) fin[f] = mag[f*T+t];
		linear(enc, fin, m->enc_w1, m->enc_b1, 1, n_bins, hidden); relu(enc, hidden);
		linear(enc2, enc, m->enc_w2, m->enc_b2, 1, hidden, hidden); relu(enc2, hidden);
		gru_step(h0, enc2, hidden, hidden, m->gru_wih[0], m->gru_whh[0], m->gru_bih[0], m->gru_bhh[0]);
		gru_step(h1, h0, hidden, hidden, m->gru_wih[1], m->gru_whh[1], m->gru_bih[1], m->gru_bhh[1]);
		linear(dec2, h1, m->dec_w1, m->dec_b1, 1, hidden, hidden); relu(dec2, hidden);
		linear(dec, dec2, m->dec_w2, m->dec_b2, 1, hidden, n_bins); sigmoid_inplace(dec, n_bins);
		for (int f = 0; f < n_bins; f++) mask[f*T+t] = dec[f];
	}

	job->output = calloc(n, sizeof(float));
	float *ws = calloc(n, sizeof(float));
	for (int t = 0; t < T; t++) {
		for (int f = 0; f < n_bins; f++) {
			float mv = mask[f*T+t];
			fb[f] = (cpx){ ph_re[f*T+t]*mv, ph_im[f*T+t]*mv };
		}
		for (int f = 1; f < n_bins-1; f++)
			fb[n_fft-f] = (cpx){ fb[f].re, -fb[f].im };
		fft(fb, n_fft, 1);
		int st = t * hop;
		for (int i = 0; i < n_fft && st+i < n; i++) {
			job->output[st+i] += fb[i].re * win[i];
			ws[st+i] += win[i] * win[i];
		}
	}
	for (int i = 0; i < n; i++) if (ws[i] > 1e-8f) job->output[i] /= ws[i];

	free(win); free(mag); free(ph_re); free(ph_im); free(fb);
	free(h0); free(h1); free(enc); free(enc2); free(dec); free(dec2);
	free(fin); free(mask); free(ws);
	return NULL;
}

/* ============ spatializer (simplified from spatialize.c) ============ */

typedef struct { float b0,b1,b2,a1,a2,x1,x2,y1,y2; } Biquad;

static void bq_hishelf(Biquad *bq, double freq, double gain_db, double sr)
{
	double A = pow(10.0, gain_db/40.0), w0 = 2.0*M_PI*freq/sr;
	double alpha = sin(w0)/2.0*sqrt((A+1.0/A)+2.0), cosw = cos(w0);
	double a0 = (A+1)-(A-1)*cosw+2*sqrt(A)*alpha;
	bq->b0=(float)((A*((A+1)+(A-1)*cosw+2*sqrt(A)*alpha))/a0);
	bq->b1=(float)((-2*A*((A-1)+(A+1)*cosw))/a0);
	bq->b2=(float)((A*((A+1)+(A-1)*cosw-2*sqrt(A)*alpha))/a0);
	bq->a1=(float)((2*((A-1)-(A+1)*cosw))/a0);
	bq->a2=(float)(((A+1)-(A-1)*cosw-2*sqrt(A)*alpha)/a0);
	bq->x1=bq->x2=bq->y1=bq->y2=0;
}

static void bq_lowpass(Biquad *bq, double freq, double q, double sr)
{
	double w0=2.0*M_PI*freq/sr, alpha=sin(w0)/(2.0*q), a0=1.0+alpha;
	bq->b0=(float)(((1.0-cos(w0))/2.0)/a0);
	bq->b1=(float)((1.0-cos(w0))/a0);
	bq->b2=bq->b0;
	bq->a1=(float)((-2.0*cos(w0))/a0);
	bq->a2=(float)((1.0-alpha)/a0);
	bq->x1=bq->x2=bq->y1=bq->y2=0;
}

static inline float bq_tick(Biquad *bq, float x)
{
	float y = bq->b0*x + bq->b1*bq->x1 + bq->b2*bq->x2
	        - bq->a1*bq->y1 - bq->a2*bq->y2;
	bq->x2=bq->x1; bq->x1=x; bq->y2=bq->y1; bq->y1=y;
	return y;
}

/* ============ main ============ */

int main(int argc, char **argv)
{
	const char *outfile = NULL;
	int duration = 0; /* 0 = full track */
	const char *model_dir = NULL;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-o") && i+1 < argc) outfile = argv[++i];
		else if (!strcmp(argv[i], "-t") && i+1 < argc) duration = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-m") && i+1 < argc) model_dir = argv[++i];
		else if (argv[i][0] != '-') {
			/* input file — remaining processing below */
			break;
		}
	}

	const char *input_path = argv[argc-1];
	if (!input_path || !outfile) {
		fprintf(stderr, "usage: %s -o out.wav [-t seconds] [-m model_dir] input\n", argv[0]);
		return 1;
	}
	if (!model_dir) model_dir = "micro/checkpoints";

	/* load input */
	SF_INFO info = {0};
	SNDFILE *sf = sf_open(input_path, SFM_READ, &info);
	if (!sf) { fprintf(stderr, "can't open %s\n", input_path); return 1; }

	int n_samples = (duration > 0) ? duration * info.samplerate : (int)info.frames;
	if (n_samples > (int)info.frames) n_samples = (int)info.frames;

	float *mono = calloc(n_samples, sizeof(float));
	float *stereo_orig = calloc(n_samples * 2, sizeof(float));

	if (info.channels == 1) {
		sf_readf_float(sf, mono, n_samples);
		for (int i = 0; i < n_samples; i++) {
			stereo_orig[i*2] = mono[i];
			stereo_orig[i*2+1] = mono[i];
		}
	} else {
		float *tmp = calloc(n_samples * info.channels, sizeof(float));
		sf_readf_float(sf, tmp, n_samples);
		for (int i = 0; i < n_samples; i++) {
			stereo_orig[i*2] = tmp[i*info.channels];
			stereo_orig[i*2+1] = (info.channels >= 2) ? tmp[i*info.channels+1] : tmp[i*info.channels];
			mono[i] = (stereo_orig[i*2] + stereo_orig[i*2+1]) * 0.5f;
		}
		free(tmp);
	}
	sf_close(sf);

	int sr = info.samplerate;
	float dur = (float)n_samples / sr;
	printf("loaded %.1fs at %dHz\n", dur, sr);

	/* load 4 models */
	char path[512];
	const char *names[] = { "vocals", "drums", "bass", "other" };
	MicroModel *models[4];
	for (int i = 0; i < 4; i++) {
		snprintf(path, sizeof(path), "%s/%s.bin", model_dir, names[i]);
		models[i] = load_micro(path);
		if (!models[i]) { fprintf(stderr, "can't load %s\n", path); return 1; }
	}

	/* separate in parallel */
	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);

	SepJob jobs[4];
	pthread_t threads[4];
	for (int i = 0; i < 4; i++) {
		jobs[i] = (SepJob){ .model = models[i], .input = mono, .n_samples = n_samples };
		pthread_create(&threads[i], NULL, separate_thread, &jobs[i]);
	}
	for (int i = 0; i < 4; i++)
		pthread_join(threads[i], NULL);

	clock_gettime(CLOCK_MONOTONIC, &t1);
	float sep_time = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9f;
	printf("separated in %.1fs (%.0fx realtime)\n", sep_time, dur/sep_time);

	/* auto-glue: compare sum of stems to original */
	float *stem_sum = calloc(n_samples, sizeof(float));
	for (int i = 0; i < 4; i++)
		for (int s = 0; s < n_samples; s++)
			stem_sum[s] += jobs[i].output[s];

	float sig_pow = 0, err_pow = 0;
	for (int s = 0; s < n_samples; s++) {
		sig_pow += mono[s] * mono[s];
		float e = mono[s] - stem_sum[s];
		err_pow += e * e;
	}
	float ser_db = (err_pow > 1e-10f) ? 10.0f * log10f(sig_pow / err_pow) : 100.0f;
	int glue_pct = 0;
	if (ser_db < 7) glue_pct = 55;
	else if (ser_db < 10) glue_pct = 40;
	else if (ser_db < 14) glue_pct = 30;
	else if (ser_db < 18) glue_pct = 20;
	else if (ser_db < 25) glue_pct = 10;
	printf("auto-glue: %d%%\n", glue_pct);
	free(stem_sum);

	/* spatialize: apply per-stem EQ + simple HRTF positioning + glue */
	/* positions: vocals front, drums behind, bass right, other left */
	float *out = calloc(n_samples * 2, sizeof(float));

	/* vocal highshelf EQ */
	Biquad voc_eq_l, voc_eq_r;
	bq_hishelf(&voc_eq_l, 3000.0, 3.5, (double)sr);
	bq_hishelf(&voc_eq_r, 3000.0, 3.5, (double)sr);

	/* bass lowpass EQ */
	Biquad bass_eq_l, bass_eq_r;
	bq_lowpass(&bass_eq_l, 300.0, 0.707, (double)sr);
	bq_lowpass(&bass_eq_r, 300.0, 0.707, (double)sr);

	/* stem gains: L/R panning based on azimuth (simple sin/cos pan law) */
	/* vocals: 0° (center), drums: 160° (behind right), bass: 20° (slight right), other: -60° (left) */
	float az_deg[] = { 0, 160, 20, -60 };
	float stem_gain = powf(10.0f, 2.3f / 20.0f); /* +2.3dB */
	float glue_gain = (glue_pct > 0) ? powf(10.0f, 20.0f * log10f(glue_pct / 100.0f) / 20.0f) : 0;

	for (int s = 0; s < n_samples; s++) {
		float L = 0, R = 0;
		for (int i = 0; i < 4; i++) {
			float sample = jobs[i].output[s] * stem_gain;

			/* per-stem EQ */
			if (i == 0) { /* vocals */
				sample = bq_tick(&voc_eq_l, sample); /* mono so same for both */
			} else if (i == 2) { /* bass */
				sample = bq_tick(&bass_eq_l, sample);
			}

			/* simple pan: equal power panning based on azimuth */
			float az = az_deg[i] * (float)M_PI / 180.0f;
			float pan = sinf(az) * 0.5f + 0.5f; /* 0=left, 1=right */
			L += sample * cosf(pan * (float)M_PI * 0.5f);
			R += sample * sinf(pan * (float)M_PI * 0.5f);
		}

		/* add glue (original mix at center) */
		if (glue_pct > 0) {
			L += mono[s] * glue_gain;
			R += mono[s] * glue_gain;
		}

		out[s*2] = L;
		out[s*2+1] = R;
	}

	/* loudness match: measure RMS of original vs output, apply gain */
	float orig_rms = 0, out_rms = 0;
	for (int s = 0; s < n_samples; s++) {
		orig_rms += stereo_orig[s*2]*stereo_orig[s*2] + stereo_orig[s*2+1]*stereo_orig[s*2+1];
		out_rms += out[s*2]*out[s*2] + out[s*2+1]*out[s*2+1];
	}
	orig_rms = sqrtf(orig_rms / (n_samples * 2));
	out_rms = sqrtf(out_rms / (n_samples * 2));

	if (out_rms > 1e-8f) {
		float gain = orig_rms / out_rms;
		for (int i = 0; i < n_samples * 2; i++)
			out[i] *= gain;
		printf("loudness: %.1fdB correction\n", 20.0f * log10f(gain));
	}

	/* peak limit */
	float peak = 0;
	for (int i = 0; i < n_samples * 2; i++) {
		float a = fabsf(out[i]);
		if (a > peak) peak = a;
	}
	if (peak > 0.95f) {
		float s = 0.95f / peak;
		for (int i = 0; i < n_samples * 2; i++) out[i] *= s;
	}

	/* write */
	SF_INFO oinfo = {
		.frames = n_samples, .samplerate = sr,
		.channels = 2, .format = SF_FORMAT_WAV | SF_FORMAT_FLOAT,
	};
	SNDFILE *osf = sf_open(outfile, SFM_WRITE, &oinfo);
	if (!osf) { fprintf(stderr, "can't write %s\n", outfile); return 1; }
	sf_writef_float(osf, out, n_samples);
	sf_close(osf);

	clock_gettime(CLOCK_MONOTONIC, &t1);
	float total = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9f;
	printf("done in %.1fs (%.0fx realtime) -> %s\n", total, dur/total, outfile);

	/* save stems */
	char stem_dir[512];
	snprintf(stem_dir, sizeof(stem_dir), "stemmed-%s", input_path);
	/* strip path and extension for dir name */
	const char *base = strrchr(input_path, '/');
	base = base ? base + 1 : input_path;
	char *dot = strrchr(base, '.');
	int blen = dot ? (int)(dot - base) : (int)strlen(base);
	snprintf(stem_dir, sizeof(stem_dir), "stemmed-%.*s", blen, base);
	mkdir(stem_dir, 0755);
	for (int i = 0; i < 4; i++) {
		snprintf(path, sizeof(path), "%s/%s.wav", stem_dir, names[i]);
		SF_INFO si = { .frames=n_samples, .samplerate=sr, .channels=1, .format=SF_FORMAT_WAV|SF_FORMAT_FLOAT };
		SNDFILE *ssf = sf_open(path, SFM_WRITE, &si);
		if (ssf) { sf_writef_float(ssf, jobs[i].output, n_samples); sf_close(ssf); }
	}
	printf("stems saved to %s/\n", stem_dir);

	free(mono); free(stereo_orig); free(out);
	for (int i = 0; i < 4; i++) free(jobs[i].output);
	return 0;
}
