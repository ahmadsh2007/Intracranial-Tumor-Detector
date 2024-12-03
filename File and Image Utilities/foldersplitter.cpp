#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void splitImageFolder(const std::string& inputFolder, const std::string& outputFolder, int n) {
    // Create the output folder if it doesn't exist
    fs::create_directories(outputFolder);

    // Collect all JPG files from the input folder
    std::vector<std::string> jpgFiles;
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            jpgFiles.push_back(entry.path().string());
        }
    }

    // Calculate the number of images per part
    int imagesPerPart = jpgFiles.size() / n;

    // Split the list of JPG files into n parts and process each part
    for (int i = 0; i < n; ++i) {
        // Create a subfolder for each part
        std::string subfolder = outputFolder + "/part_" + std::to_string(i + 1);
        fs::create_directories(subfolder);

        // Determine the range of files for the current part
        int startIdx = i * imagesPerPart;
        int endIdx = (i == n - 1) ? jpgFiles.size() : startIdx + imagesPerPart;

        for (int j = startIdx; j < endIdx; ++j) {
            // Read the image
            cv::Mat img = cv::imread(jpgFiles[j]);

            if (!img.empty()) {
                // Save the image to the corresponding subfolder
                std::string outputPath = subfolder + "/" + fs::path(jpgFiles[j]).filename().string();
                cv::imwrite(outputPath, img);
            } else {
                std::cerr << "Warning: Could not read image " << jpgFiles[j] << std::endl;
            }
        }
    }
}

int main() {
    // Replace 'inputFolder' and 'outputFolder' with your actual folder paths
    std::string inputFolder = "path/to/your/input/folder";
    std::string outputFolder = "path/to/your/output/folder";

    // Replace 'n' with the number of parts you want
    int n = 3;

    splitImageFolder(inputFolder, outputFolder, n);

    return 0;
}