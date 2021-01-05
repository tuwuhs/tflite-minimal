
#pragma once

#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include <dirent.h>

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

class TFLiteRunner
{
public:
  TFLiteRunner(std::string modelFilename, std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = nullptr);
  TFLiteRunner(const TFLiteRunner&) = delete;
  TFLiteRunner& operator=(const TFLiteRunner&) = delete;
  TFLiteRunner(TFLiteRunner&&) = default;
  virtual ~TFLiteRunner();

  void Close();
  std::vector<float> PredictImage(cv::Mat image);
  size_t PredictImageMax(cv::Mat image);

private:
  std::unique_ptr<tflite::FlatBufferModel> _model;
  std::unique_ptr<tflite::Interpreter> _interpreter;
};
