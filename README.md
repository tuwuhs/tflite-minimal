# TensorFlow Lite C++ minimal example

First, download TensorFlow and build TensorFlow Lite:
```sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/build_lib.sh
```

This will build the static library `libtensorflow-lite.a` in `tensorflow/lite/tools/make/gen/linux_x86_64/lib`.

Swap `build_lib.sh` with `build_rpi_lib.sh` or `build_aarch64_lib.sh` as appropriate. Check with the official TensorFlow Lite documentation for more details.

Set the `TENSORFLOW_DIR` environment variable then run make:
```sh
cd minimal
TENSORFLOW_DIR=~/tensorflow make
```

Usage:
```sh
./minimal [tflite_model]
```

## Original guide from the TensorFlow repo

> Note: here the CMakeLists.txt is set up to download a copy of TensorFlow source and compile it as a dependency before compiling the minimal example, which may not be what you want.

This example shows how you can build a simple TensorFlow Lite application.

#### Step 1. Install CMake tool

It requires CMake 3.16 or higher. On Ubuntu, you can simply run the following
command.

```sh
sudo apt-get install cmake
```

Or you can follow
[the official cmake installation guide](https://cmake.org/install/)

#### Step 2. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

#### Step 3. Create CMake build directory and run CMake tool

```sh
mkdir minimal_build
cd minimal_build
cmake ../tensorflow_src/tensorflow/lite/examples/minimal
```

#### Step 4. Build TensorFlow Lite

In the minimal_build directory,

```sh
cmake --build . -j
```
