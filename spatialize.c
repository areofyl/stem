/*
 * spatialize.c — Place WAV stems at 3D positions around the listener
 *                with early reflections and late reverb for immersion.
 *
 * Compile:  gcc -O3 -march=native -ffast-math -o spatialize spatialize.c -lsndfile -lm -lpthread
 * Usage:    ./spatialize -o out.wav vocals.wav:0:0 drums.wav:160:15 bass.wav:20:-5 other.wav:-60:5
 */

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __aarch64__
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif
#include <string.h>
#include <sndfile.h>

#define HEAD_RADIUS  0.0875
#define SPEED_SOUND  343.0

/* ---- Biquad filter ---- */
/* coefficients computed as double for precision, then stored as float for speed.
   state is float since we're processing float audio. */
typedef struct {
	float b0, b1, b2, a1, a2;
	float x1, x2, y1, y2;
} Biquad;

static void biquad_set_depth(Biquad *bq, double freq, double q, double depth, double sr)
{
	double w0 = 2.0 * M_PI * freq / sr;
	double alpha = sin(w0) / (2.0 * q);
	double a0 = 1.0 + alpha;
	double nb0 = 1.0 / a0;
	double nb1 = -2.0 * cos(w0) / a0;
	double nb2 = 1.0 / a0;
	double na1 = nb1;
	double na2 = (1.0 - alpha) / a0;
	double ab0 = na2, ab1 = na1, ab2 = 1.0;
	bq->b0 = (float)(ab0 + depth * (nb0 - ab0));
	bq->b1 = (float)(ab1 + depth * (nb1 - ab1));
	bq->b2 = (float)(ab2 + depth * (nb2 - ab2));
	bq->a1 = (float)na1;
	bq->a2 = (float)na2;
	bq->x1 = bq->x2 = bq->y1 = bq->y2 = 0.0f;
}

static void biquad_hishelf(Biquad *bq, double freq, double gain_db, double sr)
{
	double A  = pow(10.0, gain_db / 40.0);
	double w0 = 2.0 * M_PI * freq / sr;
	double S  = 1.0;
	double alpha = sin(w0) / 2.0 * sqrt((A + 1.0/A) * (1.0/S - 1.0) + 2.0);
	double cosw  = cos(w0);
	double a0 = (A+1) - (A-1)*cosw + 2*sqrt(A)*alpha;
	bq->b0 = (float)((A*((A+1) + (A-1)*cosw + 2*sqrt(A)*alpha)) / a0);
	bq->b1 = (float)((-2*A*((A-1) + (A+1)*cosw))                / a0);
	bq->b2 = (float)((A*((A+1) + (A-1)*cosw - 2*sqrt(A)*alpha)) / a0);
	bq->a1 = (float)((2*((A-1) - (A+1)*cosw))                   / a0);
	bq->a2 = (float)(((A+1) - (A-1)*cosw - 2*sqrt(A)*alpha)     / a0);
	bq->x1 = bq->x2 = bq->y1 = bq->y2 = 0.0f;
}

static void biquad_lowpass(Biquad *bq, double freq, double q, double sr)
{
	double w0 = 2.0 * M_PI * freq / sr;
	double alpha = sin(w0) / (2.0 * q);
	double a0 = 1.0 + alpha;
	bq->b0 = (float)(((1.0 - cos(w0)) / 2.0) / a0);
	bq->b1 = (float)((1.0 - cos(w0)) / a0);
	bq->b2 = bq->b0;
	bq->a1 = (float)((-2.0 * cos(w0)) / a0);
	bq->a2 = (float)((1.0 - alpha) / a0);
	bq->x1 = bq->x2 = bq->y1 = bq->y2 = 0.0f;
}

static inline __attribute__((always_inline)) float biquad_tick(Biquad *bq, float x)
{
	float y = bq->b0 * x + bq->b1 * bq->x1 + bq->b2 * bq->x2
	        - bq->a1 * bq->y1 - bq->a2 * bq->y2;
	bq->x2 = bq->x1; bq->x1 = x;
	bq->y2 = bq->y1; bq->y1 = y;
	return y;
}

#if USE_NEON
/*
 * NEON biquad: process two biquads (L and R) in parallel using float32x2.
 * both biquads must have the same coefficients (which they do for our
 * paired L/R filters since they're initialized with the same params).
 * even when coefficients differ (ILD makes gains different), this still
 * works because we load each biquad's state separately into lanes.
 */
typedef struct {
	float32x2_t b0, b1, b2, a1, a2;
	float32x2_t x1, x2, y1, y2;
} BiquadPair;

static inline void biquad_pair_init(BiquadPair *bp, Biquad *l, Biquad *r)
{
	bp->b0 = (float32x2_t){ l->b0, r->b0 };
	bp->b1 = (float32x2_t){ l->b1, r->b1 };
	bp->b2 = (float32x2_t){ l->b2, r->b2 };
	bp->a1 = (float32x2_t){ l->a1, r->a1 };
	bp->a2 = (float32x2_t){ l->a2, r->a2 };
	bp->x1 = bp->x2 = bp->y1 = bp->y2 = vdup_n_f32(0.0f);
}

static inline __attribute__((always_inline)) float32x2_t biquad_pair_tick(BiquadPair *bp, float32x2_t x)
{
	/* y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2
	   all operations are 2-wide (L and R simultaneously) */
	float32x2_t y = vmul_f32(bp->b0, x);
	y = vfma_f32(y, bp->b1, bp->x1);
	y = vfma_f32(y, bp->b2, bp->x2);
	y = vfms_f32(y, bp->a1, bp->y1);
	y = vfms_f32(y, bp->a2, bp->y2);

	bp->x2 = bp->x1; bp->x1 = x;
	bp->y2 = bp->y1; bp->y1 = y;
	return y;
}
#endif

/* ---- Fractional delay line for ITD ---- */
typedef struct {
	float buf[512];
	int   pos;
} Delay;

static void delay_init(Delay *d)
{
	memset(d->buf, 0, sizeof(d->buf));
	d->pos = 256;
}

static void delay_write(Delay *d, float x)
{
	d->buf[d->pos & 511] = x;
	d->pos++;
}

static float delay_read(Delay *d, double tap)
{
	int   whole = (int)floor(tap);
	float frac  = (float)(tap - whole);
	int i0 = (d->pos - whole - 2) & 511;
	int i1 = (d->pos - whole - 1) & 511;
	int i2 = (d->pos - whole)     & 511;
	int i3 = (d->pos - whole + 1) & 511;
	float y0 = d->buf[i0], y1 = d->buf[i1];
	float y2 = d->buf[i2], y3 = d->buf[i3];
	float c0 = y1;
	float c1 = 0.5f * (y2 - y0);
	float c2 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
	float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);
	return ((c3 * frac + c2) * frac + c1) * frac + c0;
}

/* ---- Large delay line for early reflections ---- */
#define RBUF_SIZE 4096
#define RBUF_MASK (RBUF_SIZE - 1)

typedef struct {
	float buf[RBUF_SIZE];
	int   pos;
} RingBuf;

static void ringbuf_init(RingBuf *r)
{
	memset(r->buf, 0, sizeof(r->buf));
	r->pos = 0;
}

static void ringbuf_write(RingBuf *r, float x)
{
	r->buf[r->pos & RBUF_MASK] = x;
	r->pos++;
}

static float ringbuf_tap(RingBuf *r, int delay)
{
	return r->buf[(r->pos - delay) & RBUF_MASK];
}

/* ---- FDN reverb (feedback delay network) ---- */
/* 8 delay lines mixed through a Hadamard matrix.
   way more diffuse and natural than Schroeder parallel combs. */

#define FDN_N 8

typedef struct {
	float *buf;
	int    len;
	int    pos;
} FDNDelay;

typedef struct {
	FDNDelay delays[FDN_N];
	float    feedback;
	float    damp[FDN_N];     /* per-line damping state */
	float    damp_coeff;      /* how much HF to absorb */
	float    gains_l[FDN_N];  /* how much each line contributes to L */
	float    gains_r[FDN_N];  /* same for R */
	float    wet;
} Reverb;

static void reverb_init(Reverb *rv, double sr, float wet)
{
	/* prime-ish delay lengths in ms for maximum density.
	   no two lines share common factors = less metallic ringing */
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

		/* stereo: alternate lines lean L or R with slight offset */
		float pan = (i % 2 == 0) ? 0.6f : 0.4f;
		/* add some variation so it's not just hard L/R */
		pan += (i / 2) * 0.05f - 0.075f;
		rv->gains_l[i] = 1.0f - pan;
		rv->gains_r[i] = pan;
	}
}

static void reverb_process(Reverb *rv, float dry_l, float dry_r, float *out_l, float *out_r)
{
	float input = (dry_l + dry_r) * 0.5f;

	/* read from all delay lines */
	float taps[FDN_N];
	for (int i = 0; i < FDN_N; i++)
		taps[i] = rv->delays[i].buf[rv->delays[i].pos];

	/* Hadamard-like mixing: each output is a weighted sum of all taps.
	   using a simple orthogonal pattern: +/- pairs that preserve energy */
	float mixed[FDN_N];
	float scale = 1.0f / 2.828f;  /* 1/sqrt(8) to preserve energy */
	for (int i = 0; i < FDN_N; i++) {
		mixed[i] = 0.0f;
		for (int j = 0; j < FDN_N; j++) {
			/* Hadamard sign pattern: positive if popcount(i&j) is even */
			int bits = i & j;
			int parity = 0;
			while (bits) { parity ^= 1; bits &= bits - 1; }
			mixed[i] += parity ? -taps[j] : taps[j];
		}
		mixed[i] *= scale;
	}

	/* write back into delay lines with feedback, damping, and input */
	float wet_l = 0.0f, wet_r = 0.0f;
	for (int i = 0; i < FDN_N; i++) {
		/* one-pole lowpass damping (walls absorb high frequencies) */
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
	for (int i = 0; i < FDN_N; i++)
		free(rv->delays[i].buf);
}

/* ---- Binaural model ---- */

static double wrap_pi(double a)
{
	while (a >  M_PI) a -= 2.0 * M_PI;
	while (a < -M_PI) a += 2.0 * M_PI;
	return a;
}

static double calc_itd(double az)
{
	az = wrap_pi(az);
	double a = fabs(az);
	double itd;
	if (a <= M_PI / 2.0)
		itd = (HEAD_RADIUS / SPEED_SOUND) * (a + sin(a));
	else
		itd = (HEAD_RADIUS / SPEED_SOUND) * (M_PI - a + sin(a));
	return (az >= 0.0) ? itd : -itd;
}

static void calc_ild(double az, double *gl, double *gr)
{
	az = wrap_pi(az);
	double s = sin(az);
	double atten = 10.0 * fabs(s);
	if (s >= 0.0) {
		*gr = 1.0;
		*gl = pow(10.0, -atten / 20.0);
	} else {
		*gl = 1.0;
		*gr = pow(10.0, -atten / 20.0);
	}
}

#define STEREO_SPREAD  (15.0 * M_PI / 180.0)

/* Binaural state for one virtual source */
typedef struct {
	Delay  dl_left, dl_right;
	double tap_l, tap_r;
	double gain_l, gain_r;
	Biquad shadow_l, shadow_r;
	Biquad pinna_l[3], pinna_r[3];
	Biquad bright_l, bright_r;
#if USE_NEON
	/* paired L/R biquads for NEON: shadow, pinna[3], bright = 5 pairs */
	BiquadPair neon_shadow;
	BiquadPair neon_pinna[3];
	BiquadPair neon_bright;
	float32x2_t neon_gain;
#endif
} BinauralState;

/* Early reflection: a delayed, attenuated bounce from a different angle */
#define NUM_REFLECTIONS 4

typedef struct {
	int           delay_samples;  /* pre-delay before binaural processing */
	float         gain;           /* attenuation (walls absorb energy) */
	BinauralState bin;            /* binaural state at reflection angle */
	Biquad        wall_lpf;       /* wall absorption filter */
} Reflection;

/* Per-stem processing state */
typedef struct {
	char   *filename;
	double azimuth;
	double elevation;
	float  stem_gain;   /* linear gain applied to this stem */

	float  *samples_l;
	float  *samples_r;
	int    num_frames;
	int    is_stereo;

	/* direct sound: one binaural path per input channel */
	BinauralState bin[2];

	/* early reflections (applied to mono sum of L+R) */
	Reflection refs[NUM_REFLECTIONS];
	RingBuf    ref_delay;  /* shared delay line for reading reflections */
} Stem;

static int load_stem(Stem *s, double sr)
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
	if (!s->samples_l || !s->samples_r) {
		sf_close(sf);
		return -1;
	}

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

	if ((int)info.samplerate != (int)sr)
		fprintf(stderr, "Warning: %s is %dHz (expected %d)\n",
		        s->filename, info.samplerate, (int)sr);

	return 0;
}

static void init_binaural(BinauralState *b, double az, double el, double sr)
{
	delay_init(&b->dl_left);
	delay_init(&b->dl_right);

	double itd_sec = calc_itd(az);
	double itd_samp = itd_sec * sr;
	if (itd_samp >= 0.0) {
		b->tap_l = 128.0 + fabs(itd_samp);
		b->tap_r = 128.0;
	} else {
		b->tap_l = 128.0;
		b->tap_r = 128.0 + fabs(itd_samp);
	}

	calc_ild(az, &b->gain_l, &b->gain_r);

	double shadow_sin = sin(wrap_pi(az));
	double shadow_freq_l, shadow_freq_r;
	if (shadow_sin > 0.0) {
		shadow_freq_l = sr * 0.45 * (1.0 - 0.7 * shadow_sin);
		shadow_freq_r = sr * 0.49;
	} else if (shadow_sin < 0.0) {
		shadow_freq_l = sr * 0.49;
		shadow_freq_r = sr * 0.45 * (1.0 - 0.7 * fabs(shadow_sin));
	} else {
		shadow_freq_l = sr * 0.49;
		shadow_freq_r = sr * 0.49;
	}
	biquad_lowpass(&b->shadow_l, shadow_freq_l, 0.707, sr);
	biquad_lowpass(&b->shadow_r, shadow_freq_r, 0.707, sr);

	double el_shift = 1.0 + 0.15 * sin(el);
	double pinna_freqs[3] = { 8000.0 * el_shift, 10500.0 * el_shift, 13000.0 * el_shift };
	double pinna_q[3]     = { 5.0, 4.0, 3.5 };

	double c = cos(wrap_pi(az));
	double depths[3];
	depths[0] = (c >= 0.0) ? 0.03 : 0.03 + 0.85 * (-c);
	depths[1] = (c >= 0.0) ? 0.15 * (1.0 - c) : 0.3 + 0.5 * (-c);
	depths[2] = (c >= 0.0) ? 0.0 : 0.7 * (-c);

	for (int i = 0; i < 3; i++) {
		biquad_set_depth(&b->pinna_l[i], pinna_freqs[i], pinna_q[i], depths[i], sr);
		biquad_set_depth(&b->pinna_r[i], pinna_freqs[i], pinna_q[i], depths[i], sr);
	}

	double bright_db = (c >= 0.0) ? 4.0 * c : 3.0 * c;
	biquad_hishelf(&b->bright_l, 3000.0, bright_db, sr);
	biquad_hishelf(&b->bright_r, 3000.0, bright_db, sr);

#if USE_NEON
	/* pack L/R biquad pairs for NEON processing */
	biquad_pair_init(&b->neon_shadow, &b->shadow_l, &b->shadow_r);
	for (int i = 0; i < 3; i++)
		biquad_pair_init(&b->neon_pinna[i], &b->pinna_l[i], &b->pinna_r[i]);
	biquad_pair_init(&b->neon_bright, &b->bright_l, &b->bright_r);
	b->neon_gain = (float32x2_t){ (float)b->gain_l, (float)b->gain_r };
#endif
}

static void init_stem_binaurals(Stem *s, double sr)
{
	double az = s->azimuth;
	double el = s->elevation;

	/* Direct sound */
	if (s->is_stereo) {
		init_binaural(&s->bin[0], az - STEREO_SPREAD, el, sr);
		init_binaural(&s->bin[1], az + STEREO_SPREAD, el, sr);
	} else {
		init_binaural(&s->bin[0], az, el, sr);
		init_binaural(&s->bin[1], az, el, sr);
	}

	/* Early reflections — simulate bounces off room surfaces.
	 * Each reflection arrives from a different angle than the direct sound,
	 * delayed by the extra path length, attenuated, and with HF rolloff
	 * from wall absorption. */
	ringbuf_init(&s->ref_delay);

	/* Reflection parameters: offset angle, delay ms, gain dB, wall LPF freq */
	struct { double az_off; double el_off; double ms; double db; double lpf; } rp[NUM_REFLECTIONS] = {
		{  100.0,   5.0,  11.0,  -7.0, 8000.0 },  /* right wall */
		{ -110.0,   5.0,  17.0,  -8.0, 7000.0 },  /* left wall */
		{  170.0,  25.0,  23.0, -10.0, 6000.0 },  /* back wall / ceiling */
		{    5.0, -30.0,   7.0, -11.0, 9000.0 },  /* floor */
	};

	for (int i = 0; i < NUM_REFLECTIONS; i++) {
		Reflection *r = &s->refs[i];
		r->delay_samples = (int)(rp[i].ms * sr / 1000.0);
		r->gain = (float)pow(10.0, rp[i].db / 20.0);

		double ref_az = az + rp[i].az_off * M_PI / 180.0;
		double ref_el = el + rp[i].el_off * M_PI / 180.0;
		init_binaural(&r->bin, ref_az, ref_el, sr);

		biquad_lowpass(&r->wall_lpf, rp[i].lpf, 0.707, sr);
	}
}

static inline __attribute__((always_inline)) void process_binaural(BinauralState *b, float sig, float *out_l, float *out_r)
{
	delay_write(&b->dl_left, sig);
	delay_write(&b->dl_right, sig);

	float L = delay_read(&b->dl_left,  b->tap_l);
	float R = delay_read(&b->dl_right, b->tap_r);

#if USE_NEON
	/* pack L/R into a 2-lane vector, run all 5 biquad pairs as SIMD.
	   each vfma does both channels in one instruction. */
	float32x2_t lr = (float32x2_t){ L, R };
	lr = vmul_f32(lr, b->neon_gain);
	lr = biquad_pair_tick(&b->neon_shadow, lr);
	lr = biquad_pair_tick(&b->neon_pinna[0], lr);
	lr = biquad_pair_tick(&b->neon_pinna[1], lr);
	lr = biquad_pair_tick(&b->neon_pinna[2], lr);
	lr = biquad_pair_tick(&b->neon_bright, lr);
	*out_l = vget_lane_f32(lr, 0);
	*out_r = vget_lane_f32(lr, 1);
#else
	L *= (float)b->gain_l;
	R *= (float)b->gain_r;

	L = biquad_tick(&b->shadow_l, L);
	R = biquad_tick(&b->shadow_r, R);

	for (int band = 0; band < 3; band++) {
		L = biquad_tick(&b->pinna_l[band], L);
		R = biquad_tick(&b->pinna_r[band], R);
	}

	L = biquad_tick(&b->bright_l, L);
	R = biquad_tick(&b->bright_r, R);

	*out_l = L;
	*out_r = R;
#endif
}

/* thread job for parallel stem processing */
typedef struct {
	Stem  *stem;
	float *buf;
	int    max_frames;
} StemJob;

static void *stem_worker(void *arg)
{
	StemJob *j = (StemJob *)arg;
	Stem *s = j->stem;
	float *buf = j->buf;
	int frames = j->max_frames;

	for (int n = 0; n < frames; n++) {
		float sig_l = (n < s->num_frames) ? s->samples_l[n] * s->stem_gain : 0.0f;
		float sig_r = (n < s->num_frames) ? s->samples_r[n] * s->stem_gain : 0.0f;

		float L, R;
		float mix_l = 0.0f, mix_r = 0.0f;

		process_binaural(&s->bin[0], sig_l, &L, &R);
		mix_l += L; mix_r += R;
		process_binaural(&s->bin[1], sig_r, &L, &R);
		mix_l += L; mix_r += R;

		float mono = (sig_l + sig_r) * 0.5f;
		ringbuf_write(&s->ref_delay, mono);

		for (int r = 0; r < NUM_REFLECTIONS; r++) {
			Reflection *ref = &s->refs[r];
			float delayed = ringbuf_tap(&s->ref_delay, ref->delay_samples);
			float filtered = biquad_tick(&ref->wall_lpf, delayed) * ref->gain;
			process_binaural(&ref->bin, filtered, &L, &R);
			mix_l += L; mix_r += R;
		}

		buf[n * 2 + 0] = mix_l;
		buf[n * 2 + 1] = mix_r;
	}
	return NULL;
}

static void usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s -o output.wav stem1.wav:az:el [stem2.wav:az:el ...]\n"
		"\n"
		"  az  = azimuth in degrees (0=front, 90=right, -90=left, 180=behind)\n"
		"  el  = elevation in degrees (0=ear level, positive=above)\n"
		"\n"
		"Example:\n"
		"  %s -o out.wav vocals.wav:0:0 drums.wav:160:15 bass.wav:20:-5 other.wav:-60:5\n",
		prog, prog);
}

static int parse_stem_arg(const char *arg, char **file, double *az, double *el, double *gain_db)
{
	/* format: filename:azimuth:elevation[:gain_db] */
	*gain_db = 0.0;

	/* Count colons from the end to find fields */
	const char *colons[4];
	int ncolons = 0;
	for (const char *p = arg + strlen(arg) - 1; p > arg && ncolons < 4; p--) {
		if (*p == ':') colons[ncolons++] = p;
	}

	if (ncolons < 2) return -1;

	const char *p_el, *p_az, *p_gain = NULL;

	if (ncolons >= 3) {
		/* Could be file:az:el:gain or file-with-colon:az:el */
		/* Try 4-field parse: check if the 3rd-from-end colon gives valid numbers */
		p_gain = colons[0];  /* last colon -> gain */
		p_el   = colons[1];  /* second-to-last -> el */
		p_az   = colons[2];  /* third-to-last -> az */

		/* Validate: az and el fields should look like numbers */
		char *end;
		strtod(p_az + 1, &end);
		if (end != p_el) {
			/* 3-field: file-with-colon:az:el */
			p_gain = NULL;
			p_el = colons[0];
			p_az = colons[1];
		}
	} else {
		p_el = colons[0];
		p_az = colons[1];
	}

	int flen = (int)(p_az - arg);
	*file = malloc(flen + 1);
	memcpy(*file, arg, flen);
	(*file)[flen] = '\0';

	*az = atof(p_az + 1) * M_PI / 180.0;
	*el = atof(p_el + 1) * M_PI / 180.0;
	if (p_gain) *gain_db = atof(p_gain + 1);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 4) {
		usage(argv[0]);
		return 1;
	}

	const char *outfile = NULL;
	float reverb_wet = 0.12f;
	int stem_start = 1;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
			outfile = argv[++i];
			stem_start = i + 1;
		} else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
			reverb_wet = (float)atof(argv[++i]);
			stem_start = i + 1;
		}
	}

	if (!outfile) {
		fprintf(stderr, "Missing -o output.wav\n");
		return 1;
	}

	int nstem = argc - stem_start;
	if (nstem < 1) {
		fprintf(stderr, "No stems specified\n");
		return 1;
	}

	Stem *stems = calloc(nstem, sizeof(Stem));

	for (int i = 0; i < nstem; i++) {
		double az, el, gain_db;
		char *file;
		if (parse_stem_arg(argv[stem_start + i], &file, &az, &el, &gain_db) < 0) {
			fprintf(stderr, "Bad stem arg: %s (expected file:az:el[:gain_db])\n",
			        argv[stem_start + i]);
			return 1;
		}
		stems[i].filename  = file;
		stems[i].azimuth   = az;
		stems[i].elevation = el;
		stems[i].stem_gain = (float)pow(10.0, gain_db / 20.0);
		printf("  stem %d: %s at az=%.0f° el=%.0f°%s\n",
		       i, file, az * 180.0 / M_PI, el * 180.0 / M_PI,
		       gain_db != 0.0 ? "" : "");
		if (gain_db != 0.0)
			printf("           gain=%.1fdB\n", gain_db);
	}

	int max_frames = 0;
	int sample_rate = 0;

	for (int i = 0; i < nstem; i++) {
		SF_INFO info = {0};
		SNDFILE *sf = sf_open(stems[i].filename, SFM_READ, &info);
		if (sf) {
			if (sample_rate == 0) sample_rate = info.samplerate;
			sf_close(sf);
		}
	}
	if (sample_rate == 0) sample_rate = 44100;

	for (int i = 0; i < nstem; i++) {
		if (load_stem(&stems[i], (double)sample_rate) < 0)
			return 1;
		if (stems[i].num_frames > max_frames)
			max_frames = stems[i].num_frames;
	}

	for (int i = 0; i < nstem; i++)
		init_stem_binaurals(&stems[i], (double)sample_rate);

	/* Global late reverb */
	Reverb reverb;
	reverb_init(&reverb, (double)sample_rate, reverb_wet);

	float *out = calloc(max_frames * 2, sizeof(float));
	if (!out) { fprintf(stderr, "alloc failed\n"); return 1; }

	/* per-stem thread data */
	StemJob *jobs = calloc(nstem, sizeof(StemJob));
	for (int i = 0; i < nstem; i++) {
		jobs[i].stem = &stems[i];
		jobs[i].buf = calloc(max_frames * 2, sizeof(float));
		jobs[i].max_frames = max_frames;
	}

	/* launch one thread per stem */
	pthread_t *threads = calloc(nstem, sizeof(pthread_t));
	for (int i = 0; i < nstem; i++)
		pthread_create(&threads[i], NULL, stem_worker, &jobs[i]);
	for (int i = 0; i < nstem; i++)
		pthread_join(threads[i], NULL);
	free(threads);

	/* merge stem buffers + reverb + peak in one pass */
	float peak = 0.0f;
	for (int n = 0; n < max_frames; n++) {
		float mix_l = 0.0f, mix_r = 0.0f;
		for (int i = 0; i < nstem; i++) {
			mix_l += jobs[i].buf[n * 2 + 0];
			mix_r += jobs[i].buf[n * 2 + 1];
		}

		float rev_l, rev_r;
		reverb_process(&reverb, mix_l, mix_r, &rev_l, &rev_r);

		out[n * 2 + 0] = rev_l;
		out[n * 2 + 1] = rev_r;

		float al = fabsf(rev_l), ar = fabsf(rev_r);
		if (al > peak) peak = al;
		if (ar > peak) peak = ar;
	}

	for (int i = 0; i < nstem; i++)
		free(jobs[i].buf);
	free(jobs);

	if (peak > 0.95f) {
		float scale = 0.95f / peak;
		for (int i = 0; i < max_frames * 2; i++)
			out[i] *= scale;
		printf("  normalized (peak was %.2f)\n", peak);
	}

	/* Write output */
	SF_INFO sfinfo = {
		.frames     = max_frames,
		.samplerate = sample_rate,
		.channels   = 2,
		.format     = SF_FORMAT_WAV | SF_FORMAT_FLOAT,
	};

	SNDFILE *sf = sf_open(outfile, SFM_WRITE, &sfinfo);
	if (!sf) {
		fprintf(stderr, "Can't write %s: %s\n", outfile, sf_strerror(NULL));
		return 1;
	}

	sf_writef_float(sf, out, max_frames);
	sf_close(sf);

	printf("Wrote %d frames (%.1fs) to %s\n",
	       max_frames, (double)max_frames / sample_rate, outfile);

	for (int i = 0; i < nstem; i++) {
		free(stems[i].filename);
		free(stems[i].samples_l);
		free(stems[i].samples_r);
	}
	free(stems);
	free(out);
	reverb_free(&reverb);
	return 0;
}
