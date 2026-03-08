CC      = gcc
CFLAGS  = -O3 -march=native -ffast-math -Wall -Wno-unused-result
LDFLAGS = -lm
SRC     = src/etok.c
BIN     = etok

.PHONY: all test bench install clean

all: $(BIN)
	@echo "  Built: $(BIN)  —  run ./etok --help"

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

debug: $(SRC)
	$(CC) -O0 -g -fsanitize=address,undefined -o $(BIN)_debug $< $(LDFLAGS)

test: $(BIN)
	python3 tests/test_etok.py

bench: $(BIN)
	python3 tests/bench.py

install: $(BIN)
	install -m755 $(BIN) /usr/local/bin/etok
	@echo "Installed → /usr/local/bin/etok"

clean:
	rm -f $(BIN) $(BIN)_debug *.json
