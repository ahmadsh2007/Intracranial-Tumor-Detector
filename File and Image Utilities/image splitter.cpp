#include <iostream>
#include <filesystem>
#include <vector>
#include <random>
#include <string>

namespace fs = std::filesystem;

void splitImages(const std::string& inputFolder, const std::string& outputFolder1, const std::string& outputFolder2, double splitRatio = 0.7) {
    // Create output folders if they don't exist
    fs::create_directories(outputFolder1);
    fs::create_directories(outputFolder2);

    // Collect all image files from the input folder
    std::vector<std::string> imageFiles;
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".gif" || ext == ".bmp" || ext == ".tiff") {
                imageFiles.push_back(entry.path().string());
            }
        }
    }

    // Calculate the number of images for each folder based on the split ratio
    int numImagesFolder1 = static_cast<int>(imageFiles.size() * splitRatio);
    int numImagesFolder2 = imageFiles.size() - numImagesFolder1;

    // Randomly shuffle the image files
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(imageFiles.begin(), imageFiles.end(), g);

    // Copy images to the output folders based on the split ratio
    for (size_t i = 0; i < imageFiles.size(); ++i) {
        const std::string& sourcePath = imageFiles[i];
        std::string destinationPath = (i < numImagesFolder1) 
                                      ? (outputFolder1 + "/" + fs::path(sourcePath).filename().string()) 
                                      : (outputFolder2 + "/" + fs::path(sourcePath).filename().string());

        fs::copy_file(sourcePath, destinationPath, fs::copy_options::overwrite_existing);
    }

    std::cout << "Splitting complete. " << numImagesFolder1 << " images in " << outputFolder1 
              << " and " << numImagesFolder2 << " images in " << outputFolder2 << "." << std::endl;
}

int main() {
    // Example usage
    std::string inputFolder = "/home/homam/Desktop/no tumor/";
    std::string outputFolder1 = "/home/homam/Desktop/no tumor 70%/";
    std::string outputFolder2 = "/home/homam/Desktop/no tumor 30%/";
    double splitRatio = 0.7;

    splitImages(inputFolder, outputFolder1, outputFolder2, splitRatio);

    return 0;
}
