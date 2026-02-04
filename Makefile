CXX      := mpicxx
ISPC     := ispc
CXXFLAGS := -O3 -std=c++17 -fopenmp -mavx2 -Isrc
ISPCFLAGS := -O3 --target=avx2-i32x8 --arch=x86-64

TARGET   := benchmark_bin
FAST_TARGET := benchmark_fast
SRC_DIR  := src

all: $(TARGET) $(FAST_TARGET)

kernels.o: $(SRC_DIR)/kernels.ispc
	$(ISPC) $(ISPCFLAGS) $< -o $@

main.o: $(SRC_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

benchmark_fast.o: $(SRC_DIR)/benchmark_fast.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): main.o kernels.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(FAST_TARGET): benchmark_fast.o kernels.o
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f *.o $(TARGET) $(FAST_TARGET)