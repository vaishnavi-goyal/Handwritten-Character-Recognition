
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(
    "handwritten_digit_model.keras"
)

# Load test dataset
(_, _), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Select one test image
index = 10
image = x_test[index]

# Preprocess image
processed_image = image.astype("float32") / 255.0
processed_image = processed_image.reshape(1, 28, 28, 1)

# Predict the digit
predictions = model.predict(processed_image, verbose=0)
predicted_digit = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

# Display results
print("Actual Digit:", y_test[index])
print("Predicted Digit:", predicted_digit)
print(f"Model Confidence: {confidence:.2f}%")

plt.imshow(image, cmap="gray")
plt.title(f"Predicted: {predicted_digit}")
plt.axis("off")
plt.show()
