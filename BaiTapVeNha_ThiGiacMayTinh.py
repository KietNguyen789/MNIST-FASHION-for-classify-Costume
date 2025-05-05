import tensorflow as tf
from tensorflow import keras

# Load MNIST dataset
mnist = keras.datasets.mnist
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

# Load MNIST Fashion dataset
fashion_mnist = keras.datasets.fashion_mnist
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train_mnist = x_train_mnist / 255.0
x_test_mnist = x_test_mnist / 255.0

x_train_fashion = x_train_fashion / 255.0
x_test_fashion = x_test_fashion / 255.0

# Build and compile the model for MNIST
model_mnist = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model_mnist.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

# Train the model on MNIST dataset
model_mnist.fit(x_train_mnist, y_train_mnist, epochs=10, validation_data=(x_test_mnist, y_test_mnist))

