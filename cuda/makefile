NVCC = nvcc
OBJFILE = cuda.cu
TARGET = q1

all: q1

q1: cuda.cu
	$(NVCC) $(OBJFILE) -o $(TARGET)

clean:
	rm -f $(TARGET) *-
