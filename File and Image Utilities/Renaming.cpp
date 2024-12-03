#include <iostream>
#include <filesystem>
#include <string>
#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

void renameJpgImages(const std::string& sourceFolder, const std::string& destinationFolder) {
    // Ensure the destination folder exists, create it if necessary
    fs::create_directories(destinationFolder);

    // Counter for renaming sequentially
    int counter = 1;

    // Iterate through each file in the source folder
    for (const auto& entry : fs::directory_iterator(sourceFolder)) {
        if (entry.is_regular_file()) {
            std::string fileName = entry.path().filename().string();
            
            // Check if the file has a .jpg extension
            if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == ".jpg") {
                // Generate a new name with a sequential number
                std::string newName = std::to_string(counter) + ".jpg";

                // Create the full path for the source and destination files
                std::string sourcePath = entry.path().string();
                std::string destinationPath = destinationFolder + "/" + newName;

                // Rename the file by moving it to the destination folder
                fs::rename(sourcePath, destinationPath);

                std::cout << "Renamed: " << fileName << " -> " << newName << std::endl;

                // Increment the counter for the next file
                counter++;
            }
        }
    }
}

int main() {
    // Replace these with actual folder paths
    std::string sourceFolder = "Drop The Path Of The Folder Here";
    std::string destinationFolder = "Drop The Path Of The Folder Here";

    // Call the function to rename JPG images
    renameJpgImages(sourceFolder, destinationFolder);

    return 0;
}

namespace fs = std::filesystem;

void renameJpgImages(const std::string& sourceFolder, const std::string& destinationFolder) {
    // Ensure the destination folder exists, create it if necessary
    fs::create_directories(destinationFolder);

    // Counter for renaming sequentially
    int counter = 1;

    // Iterate through each file in the source folder
    for (const auto& entry : fs::directory_iterator(sourceFolder)) {
        if (entry.is_regular_file()) {
            std::string fileName = entry.path().filename().string();
            
            // Check if the file has a .jpg extension
            if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == ".jpg") {
                // Generate a new name with a sequential number
                std::string newName = std::to_string(counter) + ".jpg";

                // Create the full path for the source and destination files
                std::string sourcePath = entry.path().string();
                std::string destinationPath = destinationFolder + "/" + newName;

                // Rename the file by moving it to the destination folder
                fs::rename(sourcePath, destinationPath);

                std::cout << "Renamed: " << fileName << " -> " << newName << std::endl;

                // Increment the counter for the next file
                counter++;
            }
        }
    }
}

int main() {
    // Replace these with actual folder paths
    std::string sourceFolder = "Drop The Path Of The Folder Here";
    std::string destinationFolder = "Drop The Path Of The Folder Here";

    // Call the function to rename JPG images
    renameJpgImages(sourceFolder, destinationFolder);

    return 0;
}
