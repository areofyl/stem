CC ?= gcc
CFLAGS = -O3 -march=native -ffast-math -Wall
LDFLAGS = -lsndfile -lm -lpthread -lopenblas
BLAS_CFLAGS = $(shell pkg-config --cflags openblas 2>/dev/null)

all: inference stem_micro

inference: inference.c
	$(CC) $(CFLAGS) $(BLAS_CFLAGS) -o $@ $< $(LDFLAGS)

stem_micro: stem_micro.c
	$(CC) $(CFLAGS) $(BLAS_CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f inference stem_micro

.PHONY: clean
