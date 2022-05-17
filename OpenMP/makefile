ifeq ($(OS),Windows_NT)
    CXX=.
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        CXX=g++
    endif
    ifeq ($(UNAME_S),Darwin)
        CXX=g++
    endif
endif

CXX_FLAGS=-O3 -fopenmp -I./include

$(shell mkdir -p bin)

all : quadrature_omp
    
quadrature_omp : src/quadrature_omp.cpp
	$(CXX) $(CXX_FLAGS) src/quadrature_omp.cpp -o ./bin/quadrature_omp

clean : 
	rm -rf ./bin/*
