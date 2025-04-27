[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-blue.svg)](https://jupyter.org/)
[![AI](https://img.shields.io/badge/AI-Enabled-green.svg)](https://www.ibm.com/artificial-intelligence)
[![CNN](https://img.shields.io/badge/CNN-Model-blue.svg)](https://en.wikipedia.org/wiki/Convolutional_neural_network)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Enabled-green.svg)](https://www.deeplearning.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Enabled-orange.svg)](https://www.tensorflow.org/)
[![AI Medical Project](https://img.shields.io/badge/AI%20Medical%20Project-red.svg)](https://en.wikipedia.org/wiki/Artificial_intelligence_in_healthcare)
[![MSJS Team](https://img.shields.io/badge/MSJS%20Team-Lead%20Software%20Developer-blue.svg)](https://www.linkedin.com/in/ahmadshatnawi/)

## **Introduction**

Medical errors (MEs) represent one of the most critical challenges within the healthcare sector, and they can be categorized into several types. This project focuses on **diagnostic errors**. **Misdiagnosis** is one of the most common types of malpractice, with **cancer** being one of the most likely conditions to be misdiagnosed. Given the malignant and unpredictable nature of cancer, diagnostic errors could result in **irreversible life-threatening complications** or even mortality.

## **Project Overview**

The **Intracranial Tumor Detector (ICTD)** implements the revolutionary technology of **Artificial Intelligence (AI)** within **Deep Learning (DL)** and **Convolutional Neural Networks (CNN)** in image classification to ensure no tumors go undetected. 

## **Dataset and Preprocessing**

First, we built an **MRI (Magnetic Resonance Imaging) dataset**, compromising a public dataset from **Kaggle** and images from a **local hospital**. We preprocessed the images by:

- Resizing
- Normalizing
- Augmenting them

## **Model Development**

After preprocessing, we ran the images through a **Convolutional Neural Network (CNN)** that extracts and learns features from the MRI scans, classifying them into distinct categories. Initially, the model was trained to differentiate between two main groups:

- **Brain tumor**
- **No tumor**

However, to improve the diagnostic capabilities, a second model was introduced. This model not only detected the presence of a brain tumor but also classified the type of tumor. Specifically, it was trained to detect and categorize the following tumor types:

- **Meningioma** — A type of tumor that originates in the meninges, the protective layers surrounding the brain and spinal cord.
- **Glioma** — A tumor that arises from glial cells in the brain or spine, often associated with more aggressive cancer forms.
- **Pituitary Tumor** — A tumor in or near the pituitary gland, which can affect hormone levels and other vital functions.

This second model further improved the system's ability to provide more detailed insights into the tumor's nature, aiding in better diagnosis and treatment planning.

## **Graphical User Interface (GUI)**

We then created a **Graphical User Interface (GUI)** for **medical professionals** to interact with the model.

## **Results and Conclusion**

Finally, we analyzed the results and calculated **accuracy improvement** over time. An accuracy of **over 65%** was achieved. We concluded that higher accuracy could have been achieved if:

- Better processing power had been available
- Better algorithms and tools had been used

Nevertheless, this concept is **applicable and crucial for saving lives**.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License. The full license details can be found in the [LICENSE](./LICENSE) file.

Copyright (c) 2025 Ahmad Shatnawi, MSJS Team
