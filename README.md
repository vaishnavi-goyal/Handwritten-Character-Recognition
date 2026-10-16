# ✦ DigitAI — Handwritten Digit Recognition

### Turning Handwritten Digits into Intelligent Predictions with Deep Learning

DigitAI is a Machine Learning and Computer Vision project that recognizes handwritten digits from **0 to 9** using a Convolutional Neural Network (CNN). Built with Python, TensorFlow and Streamlit, the application allows users to upload an image of a handwritten digit and receive a prediction with confidence information and class-wise probability visualization.

---

## ✨ Key Features

- **Handwritten Digit Recognition:** Predicts digits from 0 to 9.
- **CNN-Based Deep Learning:** Uses a Convolutional Neural Network to learn image patterns.
- **Image Upload:** Supports PNG, JPG and JPEG images.
- **Image Preprocessing:** Converts images to grayscale, adjusts contrast, resizes and normalizes them.
- **Prediction Confidence:** Displays the probability assigned to the predicted digit.
- **Probability Visualization:** Shows the model's probabilities for all ten digit classes.
- **Interactive Web Interface:** Provides a modern, responsive Streamlit dashboard.
- **Saved Model:** Stores the trained model for reuse without retraining every time.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | CNN model development and training |
| NumPy | Numerical operations and array processing |
| Pillow (PIL) | Image processing |
| Matplotlib | Dataset visualization during training |
| Pandas | Prediction probability data |
| Streamlit | Interactive web application |
| MNIST | Handwritten digit dataset |

## 🧠 Model Architecture

The project uses a Convolutional Neural Network trained on the MNIST handwritten digit dataset.

The model includes:

1. **Input Layer:** Accepts 28 × 28 grayscale images.
2. **Convolutional Layer:** Extracts visual features from images.
3. **Max Pooling Layer:** Reduces spatial dimensions.
4. **Second Convolutional Layer:** Learns more complex patterns.
5. **Second Max Pooling Layer:** Further reduces feature-map dimensions.
6. **Flatten Layer:** Converts feature maps into a one-dimensional representation.
7. **Dense Layer:** Learns relationships between extracted features.
8. **Dropout Layer:** Helps reduce overfitting during training.
9. **Output Layer:** Produces probabilities for the ten digit classes.

The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

## 📊 Dataset

This project uses the **MNIST handwritten digit dataset**, which contains 70,000 grayscale images of handwritten digits.

- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10 (digits 0–9)

Dataset source: [TensorFlow MNIST Dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist)

## 📁 Project Structure

```text
Handwritten_Character_Recognition/
│
├── venv/                          # Virtual environment
├── train_model.py                 # CNN training and evaluation
├── test_model.py                  # Prediction testing
├── app.py                         # Streamlit web application
├── handwritten_digit_model.keras  # Trained model
├── requirements.txt               # Required libraries
├── README.md                      # Project documentation
└── screenshots/                   # Application screenshots
```

The `venv/` directory is local to your computer and should not be uploaded to GitHub.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Handwritten_Character_Recognition
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

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

If the trained model file is not already available, run:

```bash
python train_model.py
```

The training script downloads the MNIST dataset when needed, trains the CNN, evaluates it on the test dataset and saves the model as `handwritten_digit_model.keras`.

### 5. Test a prediction

```bash
python test_model.py
```

### 6. Launch the web application

Ensure `handwritten_digit_model.keras` is in the same directory as `app.py`, then run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 💡 How to Use

1. Launch the DigitAI application.
2. Upload a clear image containing a single handwritten digit.
3. Click **Recognize My Digit**.
4. View the predicted digit and model confidence.
5. Explore the probability chart to compare the model's predictions across all ten classes.

**Tip:** Use a large, clearly written digit on a plain background for better results.

## 📈 Evaluation

The training script evaluates the model on the MNIST test dataset and prints the test accuracy.

**Test Accuracy:** Add the actual percentage reported by your model after training.

Do not enter an estimated or unverified accuracy value.

## 🔮 Future Enhancements

- Extend recognition to handwritten alphabets using an appropriate dataset such as EMNIST.
- Improve image preprocessing for real-world handwriting.
- Add prediction history and downloadable results.
- Explore model performance using a confusion matrix and per-class evaluation.
- Deploy the application online for public access.

## ⚠️ Limitations

- The current model recognizes digits, not complete words or sentences.
- Performance on real-world photographs may differ from performance on MNIST test images.
- Unusual handwriting, image noise, shadows and incorrect cropping may affect predictions.
- The displayed confidence represents the model's estimated class probability and does not guarantee correctness.

## 👩‍💻 Author

**Vaishnavi Goyal**

Machine Learning | Deep Learning | Computer Vision

## 📄 License

This project is intended for educational and portfolio purposes. Add a suitable open-source license if you decide to distribute the project publicly.

---

⭐ If you find this project useful, consider starring the repository!
