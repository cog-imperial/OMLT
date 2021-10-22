import tensorflow as tf
import tf2onnx
from pyomo.common.fileutils import this_file_dir
from tensorflow.keras import datasets, layers, models


def train_simple_cnn():
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(32, (7, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10))

    model.summary()

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    _history = model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
    )

    spec = tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input")
    tf2onnx.convert.from_keras(
        model,
        input_signature=(spec,),
        output_path=this_file_dir() + "/models/simple_cnn.onnx",
    )
