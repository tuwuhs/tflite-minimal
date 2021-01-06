
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

  // Create a context for Edge TPU, then bind with the interpreter
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = 
    edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

  auto runner = TFLiteRunner(model_filename, edgetpu_context);

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
      cv::Mat image = cv::imread(image_full_path.c_str(), cv::IMREAD_COLOR);
      auto result = runner.PredictImageMax(image);
      printf("Prediction took: %f ms\n", runner.GetLastPredictionTimeSeconds() * 1000);

      if (result != i) {
        failed += 1;
        printf("Prediction: %s   actual label: %s", labels[result].c_str(), actual_label.c_str());
      }
    }
  }

  printf("Total Images: %ld     Failed Predictions: %ld      %.2f%% Successfully Predicted\n",
    total, failed, (1.0 - 1.0*failed/total)*100.0);

  runner.Close();
  edgetpu_context.reset();

  return 0;
}