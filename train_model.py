import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

try:
    import tensorflowjs as tfjs
    TFJS_AVAILABLE = True
except ImportError:
    print("TensorFlow.js not available. The model will only be saved in H5 format.")
    TFJS_AVAILABLE = False

def train_mnist_model():
    # Load and preprocess the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # Define the model architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=5, validation_split=0.1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc}")

    # Make predictions on a few test images
    predictions = model.predict(test_images[:10])
    print("Sample predictions:")
    for i in range(10):
        print(f"True: {test_labels[i]}, Predicted: {np.argmax(predictions[i])}")

    # Save the model
    model_dir = 'static/model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, 'mnist_model.h5'))

    # Convert the model to TensorFlow.js format if available
    if TFJS_AVAILABLE:
        tfjs.converters.save_keras_model(model, model_dir)
    else:
        print("Model saved in H5 format only.")

if __name__ == '__main__':
    train_mnist_model()