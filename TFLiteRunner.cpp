
#include "TFLiteRunner.h"
#include "utils.h"

TFLiteRunner::TFLiteRunner(std::string modelFilename)
{
  _model = tflite::FlatBufferModel::BuildFromFile(modelFilename.c_str());
  TFLITE_MINIMAL_CHECK(_model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*_model, resolver);
  builder(&_interpreter);
  TFLITE_MINIMAL_CHECK(_interpreter != nullptr);

  auto ret = _interpreter->AllocateTensors();
  TFLITE_MINIMAL_CHECK(ret == kTfLiteOk);
}

TFLiteRunner::~TFLiteRunner()
{
}

std::vector<float> TFLiteRunner::PredictImage(cv::Mat image)
{
  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
  // float* input = interpreter->typed_input_tensor<float>(0);
  const std::vector<int> inputs = _interpreter->inputs();
  const std::vector<int> outputs = _interpreter->outputs();

  // printf("Inputs: ");
  // for (auto i: inputs) {
  //   printf("%d ", i);
  // }
  // printf("\n");

  // printf("Outputs: ");
  // for (auto o: outputs) {
  //   printf("%d ", o);
  // }
  // printf("\n");

  int input = inputs[0];
  TfLiteType input_type = _interpreter->tensor(input)->type;
  TfLiteIntArray* input_dims = _interpreter->tensor(input)->dims;
  int wanted_height = input_dims->data[1];
  int wanted_width = input_dims->data[2];
  int wanted_channels = input_dims->data[3];
  // printf("Input dims: %d %d %d\n", wanted_height, wanted_width, wanted_channels);

  // printf("Image dims: %d %d %d\n", image.rows, image.cols, image.channels());
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  int image_height = image_rgb.rows;
  int image_width = image_rgb.cols;
  int image_channels = image_rgb.channels();
  // printf("Image dims: %d %d %d\n", image_height, image_width, image_channels);

  switch (input_type) {
    case kTfLiteFloat32:
      resize<float>(_interpreter->typed_tensor<float>(input), image_rgb.data,
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels, input_type);
      break;
    case kTfLiteInt8:
      resize<int8_t>(_interpreter->typed_tensor<int8_t>(input), image_rgb.data,
                     image_height, image_width, image_channels, wanted_height,
                     wanted_width, wanted_channels, input_type);
      break;
    case kTfLiteUInt8:
      resize<uint8_t>(_interpreter->typed_tensor<uint8_t>(input), image_rgb.data,
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels, input_type);
      break;
    // default:
    //   LOG(ERROR) << "cannot handle input type "
    //              << interpreter->tensor(input)->type << " yet";
    //   exit(-1);
  }

  // Run inference
  auto start_time = std::chrono::high_resolution_clock::now();
  TFLITE_MINIMAL_CHECK(_interpreter->Invoke() == kTfLiteOk);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  printf("Prediction took: %f ms\n", elapsed_time.count() * 1000);

  // printf("\n\n=== Post-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
  int output = outputs[0];
  TfLiteType output_type = _interpreter->tensor(output)->type;
  TfLiteIntArray* output_dims = _interpreter->tensor(output)->dims;
  int output_size = output_dims->data[output_dims->size - 1];
  // printf("Output dims: ");
  // for (size_t i = 0; i < output_dims->size; ++i) {
  //   printf("%d ", output_dims->data[i]);
  // }
  // printf("\n");

  // printf("Output: ");
  // switch (output_type) {
  //   case kTfLiteFloat32:
  //     for (size_t i = 0; i < output_size; ++i) {
  //       printf("%f ", _interpreter->typed_tensor<float>(output)[i]);
  //     }
  //     break;
  //   case kTfLiteInt8:
  //     for (size_t i = 0; i < output_size; ++i) {
  //       printf("%d ", _interpreter->typed_tensor<int8_t>(output)[i]);
  //     }
  //     break;
  //   case kTfLiteUInt8:
  //     for (size_t i = 0; i < output_size; ++i) {
  //       printf("%d ", _interpreter->typed_tensor<uint8_t>(output)[i]);
  //     }
  //     break;
  //   // default:
  //   //   LOG(ERROR) << "cannot handle output type "
  //   //              << _interpreter->tensor(output)->type << " yet";
  //   //   exit(-1);
  // }

  std::vector<float> result;
  for (size_t i = 0; i < output_size; ++i) {
    result.push_back(_interpreter->typed_tensor<float>(output)[i]);
  }

  return result;
}

size_t TFLiteRunner::PredictImageMax(cv::Mat image)
{
  auto output = PredictImage(image);

  size_t max_idx = -1;
  float max_val = std::numeric_limits<float>::min();
  for (size_t i = 0; i < output.size(); ++i) {
    auto val = output[i];
    if (val > max_val) {
      max_val = val;
      max_idx = i;
    }
  }

  return max_idx;
}
