# 🍎 Thamara: AI Fruit Ripeness Detector

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

> An intelligent computer vision system capable of classifying fruit ripeness stages using Convolutional Neural Networks (CNN).

https://github.com/user-attachments/assets/d53c095d-1983-4c17-a1ba-32628c1ef31e

## 📖 About The Project
**Thamara (ثمرة)** is a Deep Learning project designed to automate quality control in agriculture. By analyzing fruit images, the system can determine the exact ripeness stage (e.g., Unripe, Ripe, Overripe) with high accuracy. This helps in sorting processes and reducing food waste.

The project includes a full pipeline: downloading the dataset automatically, training a custom CNN model, and exposing the model via a REST API.

## ✨ Key Features

* **🧠 Custom CNN Architecture:** A lightweight and efficient Convolutional Neural Network trained on 50 Epochs.
* **📂 Automated Data Pipeline:** Uses `kagglehub` to fetch the dataset automatically without manual download.
* **🔌 REST API:** Includes `ripeness_api.py` built with **FastAPI** to integrate the model with web or mobile apps.
* **💾 Pre-trained Model:** Comes with `thamara_ripeness_best.keras` ready for immediate deployment.
* **🔍 Instant Prediction:** The `predict_ripeness.py` script allows for testing on single images instantly.

## 🛠️ Technologies Used

* **Core Framework:** [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/).
* **Backend API:** [FastAPI](https://fastapi.tiangolo.com/).
* **Language:** Python 3.
* **Image Processing:** OpenCV / PIL.
* **Data Handling:** NumPy.

## 🚀 How to Run Locally

To use Thamara on your machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TahaniAcs/Thamara_Project.git](https://github.com/TahaniAcs/Thamara_Project.git)
    cd Thamara_Project
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy fastapi uvicorn kagglehub pillow
    ```

3.  **Option A: Run Prediction (Quick Test):**
    Put an image named `test_fruit.jpg` in the folder and run:
    ```bash
    python predict_ripeness.py
    ```

4.  **Option B: Start the API Server:**
    To use the model as a backend service:
    ```bash
    python ripeness_api.py
    ```

5.  **Option C: Train from Scratch:**
    The script will automatically download the dataset and start training:
    ```bash
    python train_cnn.py
    ```

## 📸 Model Performance
The model was trained for **50 epochs** achieving high accuracy in distinguishing between Fresh, Rotten, and Unripe fruits.

## 👥 Credits
Developed by **Tahani Althobiti**.
    python download_data.py  # Download dataset first
    python train_cnn.py      # Start training
    ```

## 📸 Model Performance
## 👥 Credits
Developed by **Tahani Althobiti**.
