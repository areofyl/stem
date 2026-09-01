CC ?= gcc
CFLAGS = -O3 -march=native -ffast-math -Wall
LDFLAGS = -lsndfile -lm -lpthread

spatialize: spatialize.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f spatialize

.PHONY: clean
