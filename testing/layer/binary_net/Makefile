CC = gcc
CFLAGS = -Wall -O2 -lm
TARGET = custom_operator
OBJECTS = main.o custom_operator.o algebraic.o ste.o sybl_parser.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(CFLAGS)

main.o: main.c custom_operator.h
	$(CC) -c main.c $(CFLAGS)

custom_operator.o: custom_operator.c custom_operator.h
	$(CC) -c custom_operator.c $(CFLAGS)

algebraic.o: algebraic.c custom_operator.h
	$(CC) -c algebraic.c $(CFLAGS)

ste.o: ste.c custom_operator.h
	$(CC) -c ste.c $(CFLAGS)

sybl_parser.o: sybl_parser.c custom_operator.h
	$(CC) -c sybl_parser.c $(CFLAGS)

clean:
	rm -f $(OBJECTS) $(TARGET)
