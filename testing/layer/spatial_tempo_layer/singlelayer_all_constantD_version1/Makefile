# Compiler and flags
CC = gcc
CFLAGS = -O2 -Wall -std=c11 -g
LDLIBS = -lm

# Sources
SRCS = main_train3d_kof.c \
       convn_forward_custom.c \
       convn_backward_custom.c \
       kof_parser.c \
       formula_eval.c \
       tinyexpr.c

# Object files (auto-generated from SRCS)
OBJS = $(SRCS:.c=.o)

# Executable
TARGET = train3d_kof

# Default target
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Pattern rule for compiling .c to .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the training binary
run: all
	./$(TARGET) kernel3d.kof 30 0.01

.PHONY: all clean run

