/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "edgetpu.h"
#include "edgetpu_c.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include "utils.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

int main(int argc, char* argv[]) {
  // Query Edge TPUs
  // From tflite/cpp/examples/lstpu/lstpu.cc
  // C style
  size_t num_devices;
  std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
    edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
  
  for (size_t i = 0; i < num_devices; ++i) {
    const auto& device = devices.get()[i];
    printf("%ld %s %s\n", i, [](auto& t){
      switch (t) {
        case EDGETPU_APEX_PCI: return "PCI";
        case EDGETPU_APEX_USB: return "USB";
        default: return "Unknown";
      }
    }(device.type), device.path);
  }
  
  // C++ style
  // auto records = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  // for (auto record: records) {
  //   printf("%s %s\n", [](auto& t){
  //     switch (t) {
  //       case edgetpu::DeviceType::kApexPci: return "PCI";
  //       case edgetpu::DeviceType::kApexUsb: return "USB";
  //       default: return "Unknown";
  //     }
  //   }(record.type), record.path.c_str());
  // }


  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model> <image file>\n");
    return 1;
  }
  const char* filename = argv[1];
  const char* image_filename = argv[2];

  auto start_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time;

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);
  elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  printf("[%8.2f ms] Done loading model\n", elapsed_time.count() * 1000);

  // Register Edge TPU
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  printf("[%8.2f ms] Done registering Edge TPU\n", elapsed_time.count() * 1000);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);
  elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  printf("[%8.2f ms] Done building interpreter\n", elapsed_time.count() * 1000);

  // Create a context for Edge TPU, then bind with the interpreter
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
    edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
  interpreter->SetNumThreads(1);
  elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  printf("[%8.2f ms] Done creating context for Edge TPU\n", elapsed_time.count() * 1000);

  // Allocate tensor buffers.
  // This will fail when an Edge TPU model is used but no Edge TPU devices
  // connected.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  printf("[%8.2f ms] Done allocating tensor ts\n", elapsed_time.count() * 1000);

  // printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
  // float* input = interpreter->typed_input_tensor<float>(0);
  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  printf("Inputs: ");
  for (auto i: inputs) {
    printf("%d ", i);
  }
  printf("\n");

  printf("Outputs: ");
  for (auto o: outputs) {
    printf("%d ", o);
  }
  printf("\n");

  int input = inputs[0];
  TfLiteType input_type = interpreter->tensor(input)->type;
  TfLiteIntArray* input_dims = interpreter->tensor(input)->dims;
  int wanted_height = input_dims->data[1];
  int wanted_width = input_dims->data[2];
  int wanted_channels = input_dims->data[3];
  printf("Input dims: %d %d %d\n", wanted_height, wanted_width, wanted_channels);

  elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  printf("[%8.2f ms] Start imread\n", elapsed_time.count() * 1000);

  cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
  // printf("Image dims: %d %d %d\n", image.rows, image.cols, image.channels());
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  int image_height = image_rgb.rows;
  int image_width = image_rgb.cols;
  int image_channels = image_rgb.channels();
  printf("Image dims: %d %d %d\n", image_height, image_width, image_channels);

  elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  printf("[%8.2f ms] Start resize\n", elapsed_time.count() * 1000);

  switch (input_type) {
    case kTfLiteFloat32:
      resize<float>(interpreter->typed_tensor<float>(input), image_rgb.data,
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels, input_type);
      break;
    case kTfLiteInt8:
      resize<int8_t>(interpreter->typed_tensor<int8_t>(input), image_rgb.data,
                     image_height, image_width, image_channels, wanted_height,
                     wanted_width, wanted_channels, input_type);
      break;
    case kTfLiteUInt8:
      resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), image_rgb.data,
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels, input_type);
      break;
    // default:
    //   LOG(ERROR) << "cannot handle input type "
    //              << interpreter->tensor(input)->type << " yet";
    //   exit(-1);
  }

  elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  printf("[%8.2f ms] Done resize\n", elapsed_time.count() * 1000);

  // Run inference
  for (size_t i = 0; i < 5; ++i) {
    auto start_time = std::chrono::high_resolution_clock::now();
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    printf("Prediction took: %f ms\n", elapsed_time.count() * 1000);
  }

  // printf("\n\n=== Post-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
  elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  printf("[%8.2f ms] Start reading output\n", elapsed_time.count() * 1000);

  int output = outputs[0];
  TfLiteType output_type = interpreter->tensor(output)->type;
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  int output_size = output_dims->data[output_dims->size - 1];
  printf("Output dims: ");
  for (size_t i = 0; i < output_dims->size; ++i) {
    printf("%d ", output_dims->data[i]);
  }
  printf("\n");

  printf("Output: ");
  switch (output_type) {
    case kTfLiteFloat32:
      for (size_t i = 0; i < output_size; ++i) {
        printf("%f ", interpreter->typed_tensor<float>(output)[i]);
      }
      break;
    case kTfLiteInt8:
      for (size_t i = 0; i < output_size; ++i) {
        printf("%d ", interpreter->typed_tensor<int8_t>(output)[i]);
      }
      break;
    case kTfLiteUInt8:
      for (size_t i = 0; i < output_size; ++i) {
        printf("%d ", interpreter->typed_tensor<uint8_t>(output)[i]);
      }
      break;
    // default:
    //   LOG(ERROR) << "cannot handle output type "
    //              << interpreter->tensor(output)->type << " yet";
    //   exit(-1);
  }
  printf("\n");

  elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
  printf("[%8.2f ms] Done reading output\n", elapsed_time.count() * 1000);

  interpreter.reset();
  edgetpu_context.reset();

  return 0;
}
