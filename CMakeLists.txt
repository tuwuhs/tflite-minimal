#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Builds the minimal Tensorflow Lite example.

cmake_minimum_required(VERSION 3.16)
project(minimal C CXX)

if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../tensorflow"
    ABSOLUTE
    CACHE
  )
endif()

if(NOT TENSORFLOW_BINARY_DIR)
  get_filename_component(TENSORFLOW_BINARY_DIR
    "${TENSORFLOW_SOURCE_DIR}/tflite_build"
    ABSOLUTE
    CACHE
  )
endif()

if(NOT LIBEDGETPU_DIR)
  get_filename_component(LIBEDGETPU_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../coral/libedgetpu"
    ABSOLUTE
    CACHE
  )
endif()

find_package( OpenCV REQUIRED )

# add_subdirectory(
#   "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
#   "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
#   EXCLUDE_FROM_ALL
# )

include_directories( 
  ${LIBEDGETPU_DIR}/tflite/public
  ${TENSORFLOW_SOURCE_DIR}
  ${TENSORFLOW_BINARY_DIR}/flatbuffers/include
  ${OpenCV_INCLUDE_DIRS} 
)

link_directories(
  ${TENSORFLOW_BINARY_DIR}
)

add_executable(minimal
  minimal.cc
)
target_link_libraries(minimal
  tensorflow-lite
  :libedgetpu.so.1.0
  atomic
  pthread
  ${CMAKE_DL_LIBS}
  ${OpenCV_LIBS}
)

add_executable(label_images
  label_images.cc
  TFLiteRunner.cpp
)
target_link_libraries(label_images
  tensorflow-lite
  :libedgetpu.so.1.0
  atomic
  pthread
  ${CMAKE_DL_LIBS}
  ${OpenCV_LIBS}
)

add_executable(label_images_multi
  label_images_multi.cc
)
target_link_libraries(label_images_multi
  tensorflow-lite
  :libedgetpu.so.1.0
  atomic
  pthread
  ${CMAKE_DL_LIBS}
  ${OpenCV_LIBS}
)
