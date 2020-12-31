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

#include <opencv2/opencv.hpp>

#include "edgetpu.h"
#include "edgetpu_c.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/bitmap_helpers_impl.h
template <class T>
void resize(T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, TfLiteType input_type) {
  int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);

  int base_index = 0;

  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op =
      resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  params->half_pixel_centers = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                     nullptr);

  interpreter->AllocateTensors();

  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }

  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;

  interpreter->Invoke();

  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

  for (int i = 0; i < output_number_of_pixels; i++) {
    switch (input_type) {
      case kTfLiteFloat32:
        out[i] = output[i] / 255.0;
        break;
      case kTfLiteInt8:
        out[i] = static_cast<int8_t>(output[i] - 128);
        break;
      case kTfLiteUInt8:
        out[i] = static_cast<uint8_t>(output[i]);
        break;
      default:
        break;
    }
  }
}

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

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Register Edge TPU
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Create a context for Edge TPU, then bind with the interpreter
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
    edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
  interpreter->SetNumThreads(1);

  // Allocate tensor buffers.
  // This will fail when an Edge TPU model is used but no Edge TPU devices
  // connected.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
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

  cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
  // printf("Image dims: %d %d %d\n", image.rows, image.cols, image.channels());
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  int image_height = image_rgb.rows;
  int image_width = image_rgb.cols;
  int image_channels = image_rgb.channels();
  printf("Image dims: %d %d %d\n", image_height, image_width, image_channels);

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

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  // printf("\n\n=== Post-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
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

  interpreter.reset();
  edgetpu_context.reset();

  return 0;
}
