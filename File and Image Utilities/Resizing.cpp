#include <iostream>
#include <filesystem>
#include <string>
#include <opencv2/opencv.hpp>  // OpenCV library for image processing

namespace fs = std::filesystem;

void resizeImages(const std::string& inputFolder, const std::string& outputFolder, const cv::Size& targetSize) {
    // Create the output folder if it doesn't exist
    fs::create_directories(outputFolder);

    // Iterate through each file in the input folder
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        if (entry.is_regular_file()) {
            std::string fileName = entry.path().filename().string();
            // Check if the file is an image (.jpg, .jpeg, .png)
            if (fileName.size() > 4 && 
                (fileName.substr(fileName.size() - 4) == ".jpg" || 
                fileName.substr(fileName.size() - 5) == ".jpeg" || 
                fileName.substr(fileName.size() - 4) == ".png")) {
                
                // Load the image
                cv::Mat img = cv::imread(entry.path().string());

                // Check if the image is loaded successfully
                if (img.empty()) {
                    std::cerr << "Could not open or find the image: " << fileName << std::endl;
                    continue;
                }

                // Resize the image
                cv::Mat imgResized;
                cv::resize(img, imgResized, targetSize, 0, 0, cv::INTER_LANCZOS4);

                // Create the output path
                std::string outputPath = outputFolder + "/" + fileName;

                // Save the resized image
                cv::imwrite(outputPath, imgResized);
                std::cout << "Processed and saved resized image: " << outputPath << std::endl;
            }
        }
    }
}

int main() {
    // Replace with the actual folder paths
    std::string inputFolder = "Drop the path of the folder here";
    std::string outputFolder = "Drop the path of the folder here";
    cv::Size targetSize(800, 600);  // Adjust the size as needed

    // Call the function to resize images
    resizeImages(inputFolder, outputFolder, targetSize);

    return 0;
}
