#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

void copyAndRenameImages(const std::vector<std::string>& sourceFolders, const std::string& destinationFolder) {
    // Create the destination folder if it doesn't exist
    fs::create_directories(destinationFolder);

    int globalIndex = 1; // Starting index for renaming files

    // Iterate through each source folder
    for (const auto& folder : sourceFolders) {
        // Iterate through all files in the current source folder
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_regular_file()) {
                std::string fileName = entry.path().filename().string();
                std::string destinationPath = destinationFolder + "/" + std::to_string(globalIndex++) + ".jpg";
                
                // Copy the file to the destination folder with the new name
                fs::copy(entry.path(), destinationPath, fs::copy_options::overwrite_existing);
            }
        }
    }

    std::cout << "Images merged and renamed successfully." << std::endl;
}

int main() {
    // List of source folders
    std::vector<std::string> sourceFolders = {
        "/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor/",
        "/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor 3/",
        "/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor part_1/",
        "/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor part_2/",
        "/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor part_3/",
        "/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/rename no tumor part_4/"
    };

    // Destination folder
    std::string destinationFolder = "/home/homam/Desktop/Projects/jupyter-for-tumor-ai/first ai test/data/no tumor/";

    // Call the function to copy and rename images
    copyAndRenameImages(sourceFolders, destinationFolder);

    return 0;
}
