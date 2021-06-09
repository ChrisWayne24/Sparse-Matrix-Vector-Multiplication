CC = mpicc
OBJFILE = mpi2.c
TARGET = q1

all: q1

q1: mpi2.c
	$(CC) $(OBJFILE) -o $(TARGET) -lm -O3 -fno-strict-aliasing -D_GNU_SOURCE -DLINUX -D_FILE_OFFSET_BITS=64 -std=c99 -Wall

clean:
	rm -f $(TARGET) *-
