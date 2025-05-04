# Gender Classification using Deep Learning

This project is a gender classification system that uses a Convolutional Neural Network (CNN) to predict gender from facial images. The model is deployed as a web app using **Streamlit**, and it supports both webcam input and image upload.

##  Model Overview

* **Model Type**: Convolutional Neural Network (CNN)
* **Input Size**: 64x64 RGB image
* **Output**: Binary classification - Male or Female
* **Framework**: TensorFlow / Keras
* **Dataset**: Custom dataset structured in `images/train/` and `images/test/` directories

## 📁 Project Structure

```
Gender_Classification_Project/
│
├── Gender_Classification.ipynb       # Jupyter Notebook for training and evaluation
├── Streamlit_App.py                  # Streamlit web application script
├── Gender classification by Prem.h5  # Trained model
├── images/
│   ├── train/                        # Training images
│   └── test/                         # Testing images
└── README.md                         # Project README (this file)
```

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gender-classification.git
cd gender-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```
tensorflow
opencv-python
numpy
streamlit
Pillow
```

### 3. Run the Streamlit App

```bash
streamlit run Streamlit_App.py
```

### 4. Upload or Capture an Image

Use the UI to either upload a `.jpg`, `.jpeg`, or `.png` image, or capture one from your webcam. The app will process the image and predict the gender.

## 📷 Real-Time Gender Detection (Optional)

To use real-time gender detection via webcam with OpenCV and audio feedback (Text-to-Speech), run:

```bash
python real_time_gender_speech.py
```

Ensure you have `pyttsx3` installed:

```bash
pip install pyttsx3
```

## 📝 Notebook

`Gender_Classification.ipynb` contains code for:

* Data preprocessing and augmentation
* Model architecture and compilation
* Training and evaluation
* Sample predictions

## ✅ Results

* Achieved over **X% accuracy** on the test dataset.
* Real-time predictions work smoothly with Haar cascades.

## 🛠 Tools & Libraries

* TensorFlow/Keras
* OpenCV
* Streamlit
* Numpy
* Pyttsx3 (for speech)
* PIL (for image handling)

## 📌 Future Work

* Improve accuracy with deeper CNN or transfer learning
* Add support for multi-class classification (e.g., age group)
* Enhance UI with model confidence levels

## 📃 License

This project is open-source and free to use for educational purposes.
