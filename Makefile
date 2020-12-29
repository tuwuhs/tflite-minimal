
# Adapted from tensorflow/lite/tools/make/Makefile
# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := osx
	endif
endif

HOST_ARCH := $(shell if uname -m | grep -q i[345678]86; then echo x86_32; else uname -m; fi)

# Note: set TENSORFLOW_DIR manually when running make
TENSORFLOW_DIR ?= 
TENSORFLOW_OUT_DIR ?= $(HOST_OS)_$(HOST_ARCH)

TARGET = minimal
CXXFLAGS = -Wall -Wextra -pedantic
INCLUDES = -I$(TENSORFLOW_DIR) -I$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/downloads/flatbuffers/include
LIBS = -pthread -ltensorflow-lite -ldl
LFLAGS = -L$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/gen/$(TENSORFLOW_OUT_DIR)/lib 

all: $(TARGET)

$(TARGET): $(TARGET).cc
	$(CXX) -o $(TARGET) $(TARGET).cc $(CXXFLAGS) $(INCLUDES) $(LFLAGS) $(LIBS) 

clean:
	$(RM) $(TARGET)
