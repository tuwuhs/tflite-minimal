#!/bin/sh

# Usage: copy to TensorFlow Lite CMake build directory, then run
# Tested to work on tensorflow@41802b3785a3e89f421254de6b6326de516cdb8d

if [ -f "libtensorflow-lite.a.old" ]; then
    exit
fi

mv libtensorflow-lite.a libtensorflow-lite.a.old

# https://stackoverflow.com/questions/50022318/using-cmake-to-build-a-static-library-of-static-libraries
ar -M <<EOM
    CREATE libtensorflow-lite.a
    ADDLIB libtensorflow-lite.a.old
    ADDLIB _deps/xnnpack-build/libXNNPACK.a
    ADDLIB cpuinfo/libcpuinfo.a
    ADDLIB clog/libclog.a
    ADDLIB pthreadpool/libpthreadpool.a
    ADDLIB _deps/farmhash-build/libfarmhash.a
    ADDLIB _deps/fft2d-build/libfft2d_fftsg.a
    ADDLIB _deps/fft2d-build/libfft2d_fftsg2d.a
    ADDLIB _deps/flatbuffers-build/libflatbuffers.a
    ADDLIB _deps/ruy-build/libruy.a
    SAVE
    END
EOM
