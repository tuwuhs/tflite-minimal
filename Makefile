
# Set these as needed, or set the env variable when running make
# TENSORFLOW_DIR = ../tensorflow
LIBEDGETPU_DIR = ../coral/libedgetpu

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

# Default directories if not provided above
TENSORFLOW_DIR ?= ../tensorflow
LIBEDGETPU_DIR ?= ../libedgetpu

TENSORFLOW_OUT_DIR ?= $(HOST_OS)_$(HOST_ARCH)

TARGET = minimal

CXXFLAGS = -Wall -Wextra -pedantic

##################################
## TensorFlow Lite Makefile build

# INCLUDES = \
# 	-I$(TENSORFLOW_DIR) \
# 	-I$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/downloads/flatbuffers/include \
# 	-I$(LIBEDGETPU_DIR)/tflite/public

# # Change to -ledgetpu if you have the library symlinked to libedgetpu.so
# LIBS = \
# 	-pthread \
# 	-ltensorflow-lite \
# 	-ldl \
# 	-lrt \
# 	-l:libedgetpu.so.1 \

# LFLAGS = \
# 	-L$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/gen/$(TENSORFLOW_OUT_DIR)/lib

################################
## TensorFlow Lite CMake build

INCLUDES = \
	-I$(TENSORFLOW_DIR) \
	-I$(TENSORFLOW_DIR)/tflite_build/flatbuffers/include \
	-I$(LIBEDGETPU_DIR)/tflite/public

# Assume that libtensorflow-lite.a is the repacked self-contained library
#   (uncomment the additional libraries for vanilla CMake build)
# Change to -ledgetpu if you have the library symlinked to libedgetpu.so
LIBS = \
	-pthread \
	-ltensorflow-lite \
	-ldl \
	-lrt \
	-l:libedgetpu.so.1 \
	# -lXNNPACK \
	# -lcpuinfo \
	# -lclog \
	# -lpthreadpool \
	# -lfarmhash \
	# -lfft2d_fftsg \
	# -lfft2d_fftsg2d \
	# -lflatbuffers \
	# -lruy 

LFLAGS = \
	-L$(TENSORFLOW_DIR)/tflite_build \
	# -L$(TENSORFLOW_DIR)/tflite_build/clog \
	# -L$(TENSORFLOW_DIR)/tflite_build/cpuinfo \
	# -L$(TENSORFLOW_DIR)/tflite_build/pthreadpool \
	# -L$(TENSORFLOW_DIR)/tflite_build/_deps/farmhash-build \
	# -L$(TENSORFLOW_DIR)/tflite_build/_deps/fft2d-build \
	# -L$(TENSORFLOW_DIR)/tflite_build/_deps/flatbuffers-build \
	# -L$(TENSORFLOW_DIR)/tflite_build/_deps/ruy-build \
	# -L$(TENSORFLOW_DIR)/tflite_build/_deps/xnnpack-build 



all: $(TARGET)

$(TARGET): $(TARGET).cc
	$(CXX) -o $(TARGET) $(TARGET).cc $(CXXFLAGS) $(INCLUDES) $(LFLAGS) $(LIBS) 

clean:
	$(RM) $(TARGET)
