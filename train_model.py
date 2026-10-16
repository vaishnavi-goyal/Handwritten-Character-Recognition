
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Step 2: Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Step 3: Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Step 4: Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Step 5: Train the model
model.summary()

history = model.fit(
    x_train[..., None],
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# Step 6: Evaluate on test data
test_loss, test_accuracy = model.evaluate(
    x_test[..., None], y_test, verbose=0
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Step 7: Save the trained model
model.save("handwritten_digit_model.keras")
print("Model saved successfully!")
