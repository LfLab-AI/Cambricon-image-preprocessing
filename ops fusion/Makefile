CC := cncc
TARGET := mlu.so
MLU_CFLAGS := --bang-mlu-arch=mtp_372 -O3 -std=c++11 -lstdc++
OPENCV_CFLAGS := -lopencv_core -lopencv_imgcodecs -I/usr/include/opencv4 -D__ARM_NEON__=0

gray_hflip: gray_hflip.mlu
	$(CC) -o $@ $^ $(MLU_CFLAGS) $(OPENCV_CFLAGS)

fusion: fusion.mlu
	$(CC) -o $(TARGET) $^ -shared -fPIC $(MLU_CFLAGS)
