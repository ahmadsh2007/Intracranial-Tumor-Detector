#include <iostream>
#include <opencv2/opencv.hpp>

// Main function
int main() {
    // Path to your image file
    // Note: Replace the path with the actual path to your image file
    std::string imagePath = "path/to/your/image.jpg";

    // Read the image
    cv::Mat img = cv::imread(imagePath);

    // Check if the image was successfully loaded
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    // Display the original image
    cv::imshow("Original Image", img);

    // Apply a simple blur filter
    cv::Mat imgBlurred;
    cv::blur(img, imgBlurred, cv::Size(5, 5)); // Kernel size can be adjusted

    // Display the blurred image
    cv::imshow("Blurred Image", imgBlurred);

    // Wait for a key press indefinitely
    cv::waitKey(0);

    // Destroy all windows (optional)
    cv::destroyAllWindows();

    return 0;
}
