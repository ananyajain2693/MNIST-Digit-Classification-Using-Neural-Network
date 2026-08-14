# MNIST-Digit-Classification-Using-Neural-Network
Deep learning project for handwritten digit classification using TensorFlow/Keras and the MNIST dataset, achieving ~97.1% test accuracy.
# MNIST Handwritten Digit Classification Using Neural Network

A deep learning project that recognizes handwritten digits (0–9) using a Neural Network trained on the **MNIST dataset**.

## 📌 Project Overview

This project demonstrates how to build an end-to-end handwritten digit classification system using **TensorFlow/Keras**.

The workflow includes:

* Loading and exploring the MNIST dataset
* Image normalization
* Building a Neural Network
* Training and evaluating the model
* Generating predictions
* Visualizing the confusion matrix
* Predicting digits from external images using OpenCV
* Creating an interactive predictive system

## 🎯 Objective

The objective is to train a Neural Network that can accurately classify handwritten digits from **0 to 9**.

## 📊 Dataset

The project uses the **MNIST handwritten digit dataset** provided through `keras.datasets`.

| Dataset  | Images | Image Size |
| -------- | -----: | ---------: |
| Training | 60,000 |    28 × 28 |
| Testing  | 10,000 |    28 × 28 |

Each image is a grayscale image containing a handwritten digit.

## 🧠 Model Architecture

The Neural Network consists of:

```text
Input Image (28 × 28)
        ↓
Flatten
        ↓
Dense Layer — 50 neurons — ReLU
        ↓
Dense Layer — 50 neurons — ReLU
        ↓
Output Layer — 10 neurons — Sigmoid
        ↓
Predicted Digit (0–9)
```

### Model Configuration

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Metric:** Accuracy
* **Epochs:** 10
* **Hidden Layers:** 2
* **Hidden Neurons:** 50 + 50

## 📈 Results

The model achieved:

* **Training Accuracy:** ~98.9%
* **Test Accuracy:** ~97.1%

This demonstrates that the Neural Network is able to recognize handwritten digits with high accuracy on unseen test data.

## 🔍 Confusion Matrix

A confusion matrix is generated to understand how well the model performs for each individual digit class.

```python
conf_mat = confusion_matrix(Y_test, Y_pred_labels)

plt.figure(figsize=(15, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
```

## 🖼️ Predicting a Custom Image

The project also supports predictions on external handwritten-digit images.

The input image is processed through the following pipeline:

```text
Custom Image
     ↓
Read using OpenCV
     ↓
Convert to Grayscale
     ↓
Resize to 28 × 28
     ↓
Normalize Pixel Values
     ↓
Reshape Image
     ↓
Neural Network
     ↓
Predicted Digit
```

Example:

```python
input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)

grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

input_image_resize = cv2.resize(grayscale, (28, 28))

input_image_resize = input_image_resize / 255

image_reshaped = np.reshape(input_image_resize, [1, 28, 28])

input_prediction = model.predict(image_reshaped)

input_pred_label = np.argmax(input_prediction)

print(
    'The Handwritten Digit is recognised as ',
    input_pred_label
)
```

## 🛠️ Technologies Used

* Python
* NumPy
* TensorFlow
* Keras
* OpenCV
* Matplotlib
* Seaborn
* PIL
* MNIST Dataset

## 📁 Repository Structure

```text
MNIST-Digit-Classification/
│
├── MNIST_Digit_Classification.ipynb
├── MNIST_digit.png
├── README.md
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/MNIST-Digit-Classification.git
cd MNIST-Digit-Classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## 📦 Requirements

```text
numpy
matplotlib
seaborn
opencv-python
Pillow
tensorflow
```

## ▶️ How to Run

1. Open `MNIST_Digit_Classification.ipynb`.
2. Install the required dependencies.
3. Run the notebook cells sequentially.
4. Train the Neural Network.
5. Evaluate the model on the MNIST test dataset.
6. View the confusion matrix.
7. Provide the path to a handwritten digit image to test the predictive system.

## 🚀 Key Learning Outcomes

Through this project, I learned how to:

* Work with image datasets
* Preprocess grayscale images
* Normalize pixel values
* Build Neural Networks using Keras
* Train deep learning models
* Evaluate classification performance
* Interpret confusion matrices
* Convert prediction probabilities into class labels
* Process custom images using OpenCV
* Build a simple real-world image prediction pipeline

## 🔮 Future Improvements

Potential improvements include:

* Replace the basic Neural Network with a **Convolutional Neural Network (CNN)**
* Improve custom-image preprocessing
* Add data augmentation
* Save and load the trained model
* Build a web interface using Streamlit
* Add confidence scores for predictions
* Deploy the application as an online digit-recognition service

## ⭐ Project Highlights

**97.1% test accuracy | 60,000 training images | 10-class classification | Custom image prediction**

This project provides a practical introduction to **Deep Learning, Neural Networks, Computer Vision, and Image Classification**.

---

## 👨‍💻 Author

**ANANYA PAIYA**

If you found this project useful, consider giving the repository a ⭐.
