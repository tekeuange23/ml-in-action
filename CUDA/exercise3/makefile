NVCC = nvcc
CFLAGS = -I./

all: main

main: main.cu matrix.cu
    $(NVCC) $(CFLAGS) $^ -o $@

clean:
    rm main