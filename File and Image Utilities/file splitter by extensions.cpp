#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

void splitFilesByExtension(const std::string& sourceFolder, const std::string& extension1Folder, const std::string& extension2Folder) {
    // Create folders if they don't exist
    fs::create_directories(extension1Folder);
    fs::create_directories(extension2Folder);

    // Iterate through all files in the source folder
    for (const auto& entry : fs::directory_iterator(sourceFolder)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            std::string fileName = entry.path().filename().string();

            // Check if the file has the specified extensions
            if (filePath.ends_with(".extension1")) {
                fs::rename(filePath, extension1Folder + "/" + fileName);
            } else if (filePath.ends_with(".extension2")) {
                fs::rename(filePath, extension2Folder + "/" + fileName);
            }
        }
    }
}

int main() {
    // Specify your folder paths
    std::string sourceFolder = "/path/to/source/folder";
    std::string extension1Folder = "/path/to/extension1/folder";
    std::string extension2Folder = "/path/to/extension2/folder";

    // Call the function to split files
    splitFilesByExtension(sourceFolder, extension1Folder, extension2Folder);

    return 0;
}
