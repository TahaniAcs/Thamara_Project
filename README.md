# 🍎 Thamara: AI Fruit Ripeness Detector

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)

> An intelligent computer vision system capable of classifying fruit ripeness stages using Convolutional Neural Networks (CNN).

## 📖 About The Project
**Thamara (ثمرة)** is a Deep Learning project designed to automate quality control in agriculture. By analyzing fruit images, the system can determine the exact ripeness stage (e.g., Unripe, Ripe, Overripe) with high accuracy. This helps in sorting processes and reducing food waste.

The project includes a full pipeline: downloading the dataset, training a custom CNN model, and running predictions on new images.

## ✨ Key Features

* **🧠 Custom CNN Architecture:** A lightweight and efficient Convolutional Neural Network trained from scratch.
* **📂 Automated Data Pipeline:** Includes `download_data.py` to easily fetch and prepare the dataset.
* **💾 Pre-trained Model:** Comes with `thamara_ripeness_cnn_model.h5` ready for immediate deployment.
* **🔍 Instant Prediction:** The `predict_ripeness.py` script allows for testing on single images instantly.

## 🛠️ Technologies Used

* **Core Framework:** [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/).
* **Language:** Python 3.
* **Image Processing:** OpenCV / PIL.
* **Data Handling:** NumPy.

## 🚀 How to Run Locally

To use Thamara on your machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TahaniAcs/Thamara_Project.git](https://github.com/TahaniAcs/Thamara_Project.git)
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy opencv-python
    ```

3.  **Run Prediction (Test the Model):**
    You can use the pre-trained model directly:
    ```bash
    python predict_ripeness.py
    ```

4.  *(Optional)* **Train from Scratch:**
    If you want to retrain the model on new data:
    ```bash
    python download_data.py  # Download dataset first
    python train_cnn.py      # Start training
    ```

## 📸 Model Performance
## 👥 Credits
Developed by **Tahani Althobiti**.
