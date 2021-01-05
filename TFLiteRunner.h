
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
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

class TFLiteRunner
{
public:
  TFLiteRunner(std::string modelFilename, std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = nullptr);
  TFLiteRunner(const TFLiteRunner&) = delete;
  TFLiteRunner& operator=(const TFLiteRunner&) = delete;
  TFLiteRunner(TFLiteRunner&&) = default;
  virtual ~TFLiteRunner();

  void Close();
  std::vector<float> PredictImage(cv::Mat imageBgr);
  size_t PredictImageMax(cv::Mat image);
  double GetLastPredictionTimeSeconds();

private:
  std::unique_ptr<tflite::FlatBufferModel> _model;
  std::unique_ptr<tflite::Interpreter> _interpreter;

  int _inputIdx;
  int _outputIdx;
  
  std::chrono::duration<double> _lastPredictionTime;
};
