# Compiler
CC = nvcc

# Compiler flags
LDLIBS = -lm
CFLAGS = -Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 -Xcompiler --openmp,-Wall

SRC = src

# Pick implementation: make → slim, make WITH_TUNER=1 → tuner, make WITH_TUNER_V2=1 → tuner-v2, make WITH_TUNER_V3=1 → tuner-v3
ifdef WITH_TUNER_V3
  LJ_IMPL = $(SRC)/lennard-jones-tuner-v3.cu
else ifdef WITH_TUNER_V2
  LJ_IMPL = $(SRC)/lennard-jones-tuner-v2.cu
else ifdef WITH_TUNER
  LJ_IMPL = $(SRC)/lennard-jones-tuner.cu
else
  LJ_IMPL = $(SRC)/lennard-jones.cu
endif

SRCS = $(LJ_IMPL) $(SRC)/main.c

# Header files
HDRS = $(wildcard $(SRC)/*.h)

# Object files
OBJS_1 = $(SRCS:$(SRC)/%.c=%.o)
OBJS   = $(OBJS_1:$(SRC)/%.cu=%.o)

# Executable name
TARGET = lj.out

# Build target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDLIBS)

# Compile source files to object files
%.o: $(SRC)/%.cu $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: $(SRC)/%.c $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

# Offline sweep — tunes skin for N=1000..150000, prints LJ_PRETUNE table to stdout.
# Usage: make sweep && ./lj_sweep.out > pretune_table.txt 2>sweep_progress.log
sweep: lj_sweep.out

lj_sweep.out: lennard-jones-tuner.o main_sweep.o
	$(CC) $(CFLAGS) lennard-jones-tuner.o main_sweep.o -o lj_sweep.out $(LDLIBS)

# Offline sweep using tuner-v2
# Usage: make sweep-v2 && ./lj_sweep_v2.out > pretune_table.txt 2>sweep_progress.log
sweep-v2: lj_sweep_v2.out

lennard-jones-tuner-v2.o: $(SRC)/lennard-jones-tuner-v2.cu $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

lj_sweep_v2.out: lennard-jones-tuner-v2.o main_sweep.o
	$(CC) $(CFLAGS) lennard-jones-tuner-v2.o main_sweep.o -o lj_sweep_v2.out $(LDLIBS)

# Offline sweep using tuner-v3
# Usage: make sweep-v3 && ./lj_sweep_v3.out > pretune_table.txt 2>sweep_progress.log
sweep-v3: lj_sweep_v3.out

lennard-jones-tuner-v3.o: $(SRC)/lennard-jones-tuner-v3.cu $(HDRS)
	$(CC) $(CFLAGS) -c -o $@ $<

lj_sweep_v3.out: lennard-jones-tuner-v3.o main_sweep.o
	$(CC) $(CFLAGS) lennard-jones-tuner-v3.o main_sweep.o -o lj_sweep_v3.out $(LDLIBS)

.PHONY: clean
clean:
	rm -f *.o $(TARGET) lj_sweep.out lj_sweep_v2.out lj_sweep_v3.out
