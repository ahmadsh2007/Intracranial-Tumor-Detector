#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <shlwapi.h>  // For PathFindFileName function to get file name

namespace fs = std::filesystem;

void splitImageFolder(const std::string& inputFolder, const std::string& outputFolder, int n) {
    // Create output folders if they don't exist
    for (int i = 0; i < n; ++i) {
        std::string folderPath = outputFolder + "/part_" + std::to_string(i + 1);
        fs::create_directories(folderPath);
    }

    // List all JPG files in the input folder
    std::vector<std::string> jpgFiles;
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        if (entry.is_regular_file() && 
            (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg")) {
            jpgFiles.push_back(entry.path().filename().string());
        }
    }

    // Calculate the number of images per part
    int imagesPerPart = jpgFiles.size() / n;

    // Split the list of JPG files into n parts
    std::vector<std::vector<std::string>> imageParts;
    for (int i = 0; i < n; ++i) {
        auto start = jpgFiles.begin() + i * imagesPerPart;
        auto end = (i == n - 1) ? jpgFiles.end() : start + imagesPerPart;
        imageParts.push_back(std::vector<std::string>(start, end));
    }

    // Process each part
    for (int i = 0; i < n; ++i) {
        // Copy each image to the corresponding output folder
        for (const auto& imageFile : imageParts[i]) {
            std::string inputPath = inputFolder + "/" + imageFile;
            std::string outputFolderPath = outputFolder + "/part_" + std::to_string(i + 1);
            std::string outputPath = outputFolderPath + "/" + imageFile;

            // Use Windows API to copy the file (same as shutil.copy2)
            if (!CopyFile(inputPath.c_str(), outputPath.c_str(), FALSE)) {
                std::cerr << "Failed to copy: " << inputPath << std::endl;
            } else {
                std::cout << "Copied: " << inputPath << " to " << outputPath << std::endl;
            }
        }
    }
}

int main() {
    // Replace with actual folder paths
    std::string inputFolder = "C:/Users/shatn/OneDrive/Desktop/Merged no tumor datasets - Copy";
    std::string outputFolder = "C:/Users/shatn/OneDrive/Desktop/Merged no tumor datasets splited";

    // Set the number of parts you want
    int n = 3;

    splitImageFolder(inputFolder, outputFolder, n);

    return 0;
}
