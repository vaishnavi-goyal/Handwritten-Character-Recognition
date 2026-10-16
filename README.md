# ✦ DigitAI — Handwritten Digit Recognition

### Turning Handwritten Digits into Intelligent Predictions with Deep Learning

**🚀 Live Demo:** [Try DigitAI](https://handwritten-character-recognition-65s8oxzwqruvwtaxaefmab.streamlit.app/)

**💻 GitHub Repository:** [View Source Code](https://github.com/vaishnavi-goyal/Handwritten-Character-Recognition)

DigitAI is a Machine Learning and Computer Vision project that recognizes handwritten digits from **0 to 9** using a Convolutional Neural Network (CNN). Built with Python, TensorFlow and Streamlit, the application allows users to upload an image of a handwritten digit and receive a prediction with confidence information and class-wise probability visualization.

---

## 🌟 Project Links

| Resource | Link |
|---|---|
| 🚀 Live Application | [Open DigitAI](https://handwritten-character-recognition-65s8oxzwqruvwtaxaefmab.streamlit.app/) |
| 💻 GitHub Repository | [View Source Code](https://github.com/vaishnavi-goyal/Handwritten-Character-Recognition) |

## ✨ Key Features

- **Handwritten Digit Recognition:** Predicts digits from 0 to 9.
- **CNN-Based Deep Learning:** Uses a Convolutional Neural Network to learn image patterns.
- **Image Upload:** Supports PNG, JPG and JPEG images.
- **Image Preprocessing:** Converts images to grayscale, adjusts contrast, resizes and normalizes them.
- **Prediction Confidence:** Displays the probability assigned to the predicted digit.
- **Probability Visualization:** Shows probabilities for all ten digit classes.
- **Interactive Web Interface:** Provides a modern Streamlit dashboard.
- **Saved Model:** Reuses the trained model without retraining for every prediction.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | CNN model development and training |
| NumPy | Numerical operations |
| Pillow | Image processing |
| Matplotlib | Dataset visualization |
| Pandas | Prediction probability data |
| Streamlit | Interactive web application |
| MNIST | Handwritten digit dataset |

## 🧠 Model Architecture

The project uses a Convolutional Neural Network trained on the MNIST dataset.

1. **Input Layer:** Accepts 28 × 28 grayscale images.
2. **Convolutional Layers:** Extract visual features and patterns.
3. **Max Pooling Layers:** Reduce spatial dimensions.
4. **Flatten Layer:** Converts feature maps into a one-dimensional representation.
5. **Dense Layer:** Learns relationships between extracted features.
6. **Dropout Layer:** Helps reduce overfitting during training.
7. **Output Layer:** Produces probabilities for ten digit classes.

The model uses the Adam optimizer and sparse categorical cross-entropy loss.

## 📊 Dataset

This project uses the **MNIST handwritten digit dataset**, containing 70,000 grayscale images.

- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10 (digits 0–9)

Dataset source: [TensorFlow MNIST Dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist)

## 📁 Project Structure

```text
Handwritten-Character-Recognition/
│
├── app.py
├── train_model.py
├── test_model.py
├── handwritten_digit_model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

The `venv/` virtual environment is local to your computer and should not be uploaded to GitHub.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/vaishnavi-goyal/Handwritten-Character-Recognition.git
cd Handwritten-Character-Recognition
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train the model (optional)

If the trained model file is unavailable, run:

```bash
python train_model.py
```

This trains the CNN on MNIST, evaluates it and saves the model as `handwritten_digit_model.keras`.

### 5. Test a prediction

```bash
python test_model.py
```

### 6. Launch the application

Ensure `handwritten_digit_model.keras` is available in the same directory as `app.py`, then run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 💡 How to Use

1. Open the [DigitAI Live Application](https://handwritten-character-recognition-65s8oxzwqruvwtaxaefmab.streamlit.app/).
2. Upload an image containing a single handwritten digit.
3. Click **Recognize My Digit**.
4. View the predicted digit and model confidence.
5. Explore the probability chart for all ten digit classes.

**Tip:** Use a large, clearly written digit on a plain background for better results.

## 📈 Model Evaluation

The training script evaluates the CNN on the MNIST test dataset and prints the test accuracy.

**Test Accuracy:** Add the actual percentage reported by your model after training.

Do not enter an estimated or unverified accuracy value.

## 🔮 Future Enhancements

- Extend recognition to handwritten alphabets using a dataset such as EMNIST.
- Improve preprocessing for real-world handwriting.
- Add prediction history and downloadable results.
- Evaluate performance using a confusion matrix and per-class metrics.
- Explore additional model architectures and deployment improvements.

## ⚠️ Limitations

- The current model recognizes individual digits, not complete words or sentences.
- Real-world photographs may produce different results from MNIST test images.
- Shadows, noise, unusual handwriting and incorrect cropping can affect predictions.
- Model confidence does not guarantee that a prediction is correct.

## 👩‍💻 Author

**Vaishnavi Goyal**

Machine Learning | Deep Learning | Computer Vision

- [GitHub Profile](https://github.com/vaishnavi-goyal)
- [Project Repository](https://github.com/vaishnavi-goyal/Handwritten-Character-Recognition)
- [Live Project](https://handwritten-character-recognition-65s8oxzwqruvwtaxaefmab.streamlit.app/)

---

⭐ If you find this project useful, consider starring the repository!

*Built with Python, TensorFlow, CNN and Streamlit.*
