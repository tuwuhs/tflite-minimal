
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include <dirent.h>

#include <cstdio>
#include <chrono>

#include <opencv2/opencv.hpp>

// #include "edgetpu.h"
// #include "edgetpu_c.h"
// #include "tensorflow/lite/builtin_op_data.h"
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/optional_debug_tools.h"
// #include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include "TFLiteRunner.h"

#include "utils.h"

std::vector<std::string> readLabels(std::string labels_filename)
{
  std::vector<std::string> labels;
  std::ifstream labels_file;

  labels_file.open(labels_filename);
  if (!labels_file.is_open()) {
    fprintf(stderr, "class_labels.txt not found");
    exit(1);
  }
  for (std::string line; std::getline(labels_file, line); ) {
    labels.push_back(line);
  }

  return labels;
}

std::vector<std::string> readImagesFullPath(std::string images_path)
{
  DIR* dir;
  struct dirent *ent;
  std::vector<std::string> result;

  // printf("%s\n", images_path.c_str());
  if (dir = opendir(images_path.c_str())) {
    while (ent = readdir(dir)) {
      if (ent->d_type != DT_REG)
        continue;
      
      std::stringstream image_fullpath;
      image_fullpath << images_path << '/' << ent->d_name;

      // printf("%s\n", image_fullpath.str().c_str());
      result.push_back(image_fullpath.str());
    }
    closedir(dir);
  }

  return result;
}

// int predictImage(std::unique_ptr<tflite::Interpreter>& interpreter, std::string image_filename)
// {
//   // Fill input buffers
//   // TODO(user): Insert code to fill input tensors.
//   // Note: The buffer of the input tensor with index `i` of type T can
//   // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
//   // float* input = interpreter->typed_input_tensor<float>(0);
//   const std::vector<int> inputs = interpreter->inputs();
//   const std::vector<int> outputs = interpreter->outputs();

//   // printf("Inputs: ");
//   // for (auto i: inputs) {
//   //   printf("%d ", i);
//   // }
//   // printf("\n");

//   // printf("Outputs: ");
//   // for (auto o: outputs) {
//   //   printf("%d ", o);
//   // }
//   // printf("\n");

//   int input = inputs[0];
//   TfLiteType input_type = interpreter->tensor(input)->type;
//   TfLiteIntArray* input_dims = interpreter->tensor(input)->dims;
//   int wanted_height = input_dims->data[1];
//   int wanted_width = input_dims->data[2];
//   int wanted_channels = input_dims->data[3];
//   // printf("Input dims: %d %d %d\n", wanted_height, wanted_width, wanted_channels);

//   cv::Mat image = cv::imread(image_filename.c_str(), cv::IMREAD_COLOR);
//   // printf("Image dims: %d %d %d\n", image.rows, image.cols, image.channels());
//   cv::Mat image_rgb;
//   cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
//   int image_height = image_rgb.rows;
//   int image_width = image_rgb.cols;
//   int image_channels = image_rgb.channels();
//   // printf("Image dims: %d %d %d\n", image_height, image_width, image_channels);

//   switch (input_type) {
//     case kTfLiteFloat32:
//       resize<float>(interpreter->typed_tensor<float>(input), image_rgb.data,
//                     image_height, image_width, image_channels, wanted_height,
//                     wanted_width, wanted_channels, input_type);
//       break;
//     case kTfLiteInt8:
//       resize<int8_t>(interpreter->typed_tensor<int8_t>(input), image_rgb.data,
//                      image_height, image_width, image_channels, wanted_height,
//                      wanted_width, wanted_channels, input_type);
//       break;
//     case kTfLiteUInt8:
//       resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), image_rgb.data,
//                       image_height, image_width, image_channels, wanted_height,
//                       wanted_width, wanted_channels, input_type);
//       break;
//     // default:
//     //   LOG(ERROR) << "cannot handle input type "
//     //              << interpreter->tensor(input)->type << " yet";
//     //   exit(-1);
//   }

//   // Run inference
//   auto start_time = std::chrono::high_resolution_clock::now();
//   TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
//   auto end_time = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double> elapsed_time = end_time - start_time;
//   printf("Prediction took: %f ms\n", elapsed_time.count() * 1000);

//   // printf("\n\n=== Post-invoke Interpreter State ===\n");
//   // tflite::PrintInterpreterState(interpreter.get());

//   // Read output buffers
//   // TODO(user): Insert getting data out code.
//   // Note: The buffer of the output tensor with index `i` of type T can
//   // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
//   int output = outputs[0];
//   TfLiteType output_type = interpreter->tensor(output)->type;
//   TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
//   int output_size = output_dims->data[output_dims->size - 1];
//   // printf("Output dims: ");
//   // for (size_t i = 0; i < output_dims->size; ++i) {
//   //   printf("%d ", output_dims->data[i]);
//   // }
//   // printf("\n");

//   // printf("Output: ");
//   // switch (output_type) {
//   //   case kTfLiteFloat32:
//   //     for (size_t i = 0; i < output_size; ++i) {
//   //       printf("%f ", interpreter->typed_tensor<float>(output)[i]);
//   //     }
//   //     break;
//   //   case kTfLiteInt8:
//   //     for (size_t i = 0; i < output_size; ++i) {
//   //       printf("%d ", interpreter->typed_tensor<int8_t>(output)[i]);
//   //     }
//   //     break;
//   //   case kTfLiteUInt8:
//   //     for (size_t i = 0; i < output_size; ++i) {
//   //       printf("%d ", interpreter->typed_tensor<uint8_t>(output)[i]);
//   //     }
//   //     break;
//   //   // default:
//   //   //   LOG(ERROR) << "cannot handle output type "
//   //   //              << interpreter->tensor(output)->type << " yet";
//   //   //   exit(-1);
//   // }

//   size_t max_idx = -1;
//   switch (output_type) {
//     case kTfLiteFloat32:
//       {
//         float max_val = std::numeric_limits<float>::min();
//         for (size_t i = 0; i < output_size; ++i) {
//           auto val = interpreter->typed_tensor<float>(output)[i];
//           if (val > max_val) {
//             max_val = val;
//             max_idx = i;
//           }
//         }
//       }
//       break;
//     case kTfLiteInt8:
//       {
//         int8_t max_val = std::numeric_limits<int8_t>::min();
//         for (size_t i = 0; i < output_size; ++i) {
//           auto val = interpreter->typed_tensor<int8_t>(output)[i];
//           if (val > max_val) {
//             max_val = val;
//             max_idx = i;
//           }
//         }
//       }
//       break;
//     case kTfLiteUInt8:
//       {
//         uint8_t max_val = std::numeric_limits<uint8_t>::min();
//         for (size_t i = 0; i < output_size; ++i) {
//           auto val = interpreter->typed_tensor<uint8_t>(output)[i];
//           if (val > max_val) {
//             max_val = val;
//             max_idx = i;
//           }
//         }
//       }
//       break;
//     // default:
//     //   LOG(ERROR) << "cannot handle output type "
//     //              << interpreter->tensor(output)->type << " yet";
//     //   exit(-1);
//   }

//   return max_idx;
// }

int main(int argc, char* argv[])
{
  if (argc != 3) {
    fprintf(stderr, "label_images <tflite model> <dataset path>\n");
    return 1;
  }

  auto model_filename = argv[1];
  auto folder_path = std::string(argv[2]);
  auto labels = readLabels("../class_labels.txt");

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

  auto runner = TFLiteRunner(model_filename);

  // // Load model
  // std::unique_ptr<tflite::FlatBufferModel> model =
  //     tflite::FlatBufferModel::BuildFromFile(model_filename);
  // TFLITE_MINIMAL_CHECK(model != nullptr);

  // // Register Edge TPU
  // tflite::ops::builtin::BuiltinOpResolver resolver;
  // resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

  // // Build the interpreter with the InterpreterBuilder.
  // // Note: all Interpreters should be built with the InterpreterBuilder,
  // // which allocates memory for the Intrepter and does various set up
  // // tasks so that the Interpreter can read the provided model.
  // tflite::InterpreterBuilder builder(*model, resolver);
  // std::unique_ptr<tflite::Interpreter> interpreter;
  // builder(&interpreter);
  // TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // // Create a context for Edge TPU, then bind with the interpreter
  // std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = 
  //   edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  // interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
  // interpreter->SetNumThreads(1);

  // // Allocate tensor buffers.
  // // This will fail when an Edge TPU model is used but no Edge TPU devices
  // // connected.
  // TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  size_t total = 0;
  size_t failed = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    auto actual_label = labels[i];
    std::stringstream images_path;
    images_path << folder_path;
    if (folder_path.back() != '/') {
      images_path << '/';
    }
    images_path << actual_label;

    auto images_full_path = readImagesFullPath(images_path.str());
    for (auto image_full_path: images_full_path) {
      total += 1;
      // auto result = predictImage(interpreter, image_full_path);
      cv::Mat image = cv::imread(image_full_path.c_str(), cv::IMREAD_COLOR);
      auto result = runner.PredictImageMax(image);
      
      if (result != i) {
        failed += 1;
        printf("Prediction: %s   actual label: %s", labels[result].c_str(), actual_label.c_str());
      }
    }
  }

  printf("Total Images: %ld     Failed Predictions: %ld      %.2f%% Successfully Predicted\n",
    total, failed, (1.0 - 1.0*failed/total)*100.0);

  // interpreter.reset();
  // edgetpu_context.reset();

  return 0;
}