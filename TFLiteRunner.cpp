
#include "TFLiteRunner.h"
#include "utils.h"

TFLiteRunner::TFLiteRunner(std::string modelFilename, std::shared_ptr<edgetpu::EdgeTpuContext> edgeTpuContext)
{
  _model = tflite::FlatBufferModel::BuildFromFile(modelFilename.c_str());
  TFLITE_MINIMAL_CHECK(_model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  if (edgeTpuContext != nullptr) {
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  }

  tflite::InterpreterBuilder builder(*_model, resolver);
  builder(&_interpreter);
  TFLITE_MINIMAL_CHECK(_interpreter != nullptr);

  if (edgeTpuContext != nullptr) {
    _interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgeTpuContext.get());
    _interpreter->SetNumThreads(1);
  }

  auto ret = _interpreter->AllocateTensors();
  TFLITE_MINIMAL_CHECK(ret == kTfLiteOk);

  // Assume single input single output, both float32
  const std::vector<int> inputs = _interpreter->inputs();
  const std::vector<int> outputs = _interpreter->outputs();
  _inputIdx = inputs[0];
  _outputIdx = outputs[0];

  TfLiteType inputType = _interpreter->tensor(_inputIdx)->type;
  TFLITE_MINIMAL_CHECK(inputType == kTfLiteFloat32);

  TfLiteType outputType = _interpreter->tensor(_outputIdx)->type;
  TFLITE_MINIMAL_CHECK(outputType == kTfLiteFloat32);
}

TFLiteRunner::~TFLiteRunner()
{
}

void TFLiteRunner::Close()
{
  _interpreter.reset();
}

std::vector<float> TFLiteRunner::PredictImage(cv::Mat imageBgr)
{
  TfLiteIntArray* inputDims = _interpreter->tensor(_inputIdx)->dims;
  int wantedHeight = inputDims->data[1];
  int wantedWidth = inputDims->data[2];
  int wantedChannels = inputDims->data[3];
  // printf("Input dims: %d %d %d\n", wantedHeight, wantedWidth, wantedChannels);

  // printf("Image dims: %d %d %d\n", image.rows, image.cols, image.channels());
  cv::Mat imageRgb;
  cv::cvtColor(imageBgr, imageRgb, cv::COLOR_BGR2RGB);
  int imageHeight = imageRgb.rows;
  int imageWidth = imageRgb.cols;
  int imageChannels = imageRgb.channels();
  // printf("Image dims: %d %d %d\n", imageHeight, imageWidth, imageChannels);

  auto input = _interpreter->typed_tensor<float>(_inputIdx);
  if (imageHeight == wantedHeight && imageWidth == wantedWidth && imageChannels == wantedChannels) {
    auto numberOfPixels = imageHeight * imageWidth * imageChannels;
    for (int i = 0; i < numberOfPixels; i++) {
      input[i] = imageRgb.data[i];
    }
  } else {
    resize<float>(input, imageRgb.data,
                  imageHeight, imageWidth, imageChannels, 
                  wantedHeight, wantedWidth, wantedChannels, kTfLiteFloat32);
  }

  // Run inference
  auto startTime = std::chrono::high_resolution_clock::now();
  TFLITE_MINIMAL_CHECK(_interpreter->Invoke() == kTfLiteOk);
  auto endTime = std::chrono::high_resolution_clock::now();
  _lastPredictionTime = endTime - startTime;

  // Read output buffers
  TfLiteIntArray* outputDims = _interpreter->tensor(_outputIdx)->dims;
  int outputSize = outputDims->data[outputDims->size - 1];

  std::vector<float> result;
  for (size_t i = 0; i < outputSize; ++i) {
    result.push_back(_interpreter->typed_tensor<float>(_outputIdx)[i]);
  }

  return result;
}

size_t TFLiteRunner::PredictImageMax(cv::Mat image)
{
  auto output = PredictImage(image);

  size_t maxIdx = -1;
  float maxVal = std::numeric_limits<float>::min();
  for (size_t i = 0; i < output.size(); ++i) {
    auto val = output[i];
    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }

  return maxIdx;
}

double TFLiteRunner::GetLastPredictionTimeSeconds()
{
  return _lastPredictionTime.count();
}
