CC ?= gcc
CFLAGS = -O3 -march=native -ffast-math -Wall
LDFLAGS = -lsndfile -lm -lpthread
PREFIX ?= /usr/local

all: spatialize inference

spatialize: spatialize.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

inference: inference.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

install: all
	install -Dm755 spatialize $(DESTDIR)$(PREFIX)/bin/spatialize
	install -Dm755 inference $(DESTDIR)$(PREFIX)/bin/stem-inference
	install -Dm755 stem $(DESTDIR)$(PREFIX)/bin/stem
	install -Dm755 fast_separate.py $(DESTDIR)$(PREFIX)/lib/stem/fast_separate.py
	install -Dm755 autoglue.py $(DESTDIR)$(PREFIX)/lib/stem/autoglue.py
	install -Dm644 train/model.py $(DESTDIR)$(PREFIX)/lib/stem/train/model.py
	install -Dm755 train/inference.py $(DESTDIR)$(PREFIX)/lib/stem/train/inference.py

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/spatialize
	rm -f $(DESTDIR)$(PREFIX)/bin/stem-inference
	rm -f $(DESTDIR)$(PREFIX)/bin/stem
	rm -rf $(DESTDIR)$(PREFIX)/lib/stem

clean:
	rm -f spatialize inference

.PHONY: all clean install uninstall
