CXX ?= g++
CXXFLAGS += -g -std=c++11 -fdata-sections -ffunction-sections -Wl,--gc-sections
CXXFLAGS += -static-libgcc -static-libstdc++ -Wl,-Bstatic -ljemalloc_pic
CXXFLAGS += -Wl,-Bstatic -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group -Wl,-Bdynamic -lpthread -lm -ldl
CXXFLAGS += -Wl,-Bstatic -lhptt -Lhptt/lib -Ihptt/include
CXXFLAGS += -Iargs

DEBUG ?= 1
ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG -pg -O0 -Wall -Wextra -fprofile-arcs -ftest-coverage
else
	CXXFLAGS += -DNDEBUG -Ofast -march=native -fwhole-program
endif

compile: FILE ?= test.cpp
compile:
	$(CXX) $(FILE) $(CXXFLAGS) -o $(FILE:.cpp=.out)

style: FILE ?= TAT.hpp
style:
	astyle --indent=spaces=2 --indent-namespaces --style=google --pad-comma --pad-header --align-pointer=type --align-reference=type $(FILE)
