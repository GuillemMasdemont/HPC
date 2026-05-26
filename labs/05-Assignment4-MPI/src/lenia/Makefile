# Compiler
CC = mpicc

# Compiler flags
LDLIBS = -lm
CFLAGS = -O3 -Wall

SRC=src

# Source files
SRCS+=$(wildcard $(SRC)/*.c)

# Header files
HDRS = $(wildcard $(SRC)/*.h)

# Object files
OBJS = $(SRCS:$(SRC)/%.c=%.o)

# Executable name
TARGET = lenia.out

# Build target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

# Compile source files to object files
%.o: $(SRC)/%.cu $(HDRS)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

%.o: $(SRC)/%.c $(HDRS)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<
# Clean
clean:
	rm -f $(OBJS) $(TARGET)