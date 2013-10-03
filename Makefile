LOCAL_LIBS=/usr/local
EXTRA_LIBS=/data/taipan/karher/extra_lib
override CXXFLAGS += -g -std=c++0x -fopenmp -Ofast -Wall -c -MMD -I${LOCAL_LIBS}/include -m64 -I/usr/include/eigen3 -I${EXTRA_LIBS}/include
override LDFLAGS += -g -std=c++0x -fopenmp -Ofast -L${LOCAL_LIBS}/lib64 -L${LOCAL_LIBS}/lib -L${EXTRA_LIBS}/lib -lboost_program_options -lboost_serialization -llbfgs
### Normally -Ofast instead of -O0

DIRS    := src/models/mvrnn src/common src/pugi src/models/ccaeb
SOURCES := $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cc) $(wildcard $(dir)/*.cpp))
OBJS    := $(patsubst %.cc, %.o, $(SOURCES))
OBJS    := $(patsubst %.cpp, %.o, $(OBJS))
OBJS    := $(foreach o,$(OBJS),build/$(o))

DEPFILES:= $(patsubst %.o, %.P, $(OBJS))

CXX = g++

#link the executable
all: train

train: $(OBJS) build/src/train.o
	$(CXX) -o train $^ $(LDFLAGS)

#generate dependency information and compile
build/%.o : %.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ -MF build/$*.P $<
	@sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < build/$*.P >> build/$*.P;

build/%.o : %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ -MF build/$*.P $<
	@sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < build/$*.P >> build/$*.P;

#remove all generated files
clean:
	rm -f main
	rm -rf build

#include the dependency information
-include $(DEPFILES)
