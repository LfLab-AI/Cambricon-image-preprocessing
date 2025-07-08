CC := cncc
TARGET := mlu.so
MLU_CFLAGS := --bang-mlu-arch=mtp_372 -O3 -std=c++11 -lstdc++ -llapacke

all: interface.mlu
	$(CC) -o $(TARGET) $^ -shared -fPIC $(MLU_CFLAGS)
