import keras
import numpy as np
import pytest
from conftest import get_neural_network_data
from keras.layers import Dense
from keras.models import Model, Sequential
from pyomo.common.fileutils import this_file_dir
from tensorflow.keras.optimizers import Adamax


def train_models():
    x, y, x_test = get_neural_network_data("131")
    nn = Sequential(name="keras_linear_131")
    nn.add(
        Dense(
            units=3,
            input_dim=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=42
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=43
            ),
        )
    )
    nn.add(
        Dense(
            units=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=62
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=63
            ),
        )
    )
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss="mae")
    history = nn.fit(
        x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15
    )
    nn.save(this_file_dir() + "/models/keras_linear_131")

    x, y, x_test = get_neural_network_data("131")
    nn = Sequential(name="keras_linear_131_sigmoid")
    nn.add(
        Dense(
            units=3,
            input_dim=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=42
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=43
            ),
            activation="sigmoid",
        )
    )
    nn.add(
        Dense(
            units=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=62
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=63
            ),
        )
    )
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss="mae")
    history = nn.fit(
        x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15
    )
    nn.save(this_file_dir() + "/models/keras_linear_131_sigmoid")

    x, y, x_test = get_neural_network_data("131")
    nn = Sequential(name="keras_linear_131_sigmoid_output_activation")
    nn.add(
        Dense(
            units=3,
            input_dim=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=42
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=43
            ),
            activation="sigmoid",
        )
    )
    nn.add(
        Dense(
            units=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=62
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=63
            ),
            activation="sigmoid",
        )
    )
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss="mae")
    history = nn.fit(
        x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15
    )
    nn.save(this_file_dir() + "/models/keras_linear_131_sigmoid_output_activation")

    x, y, x_test = get_neural_network_data("131")
    nn = Sequential(name="keras_linear_131_relu")
    nn.add(
        Dense(
            units=3,
            input_dim=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=42
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=43
            ),
            activation="relu",
        )
    )
    nn.add(
        Dense(
            units=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=62
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=63
            ),
        )
    )
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss="mae")
    history = nn.fit(
        x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15
    )
    nn.save(this_file_dir() + "/models/keras_linear_131_relu")

    x, y, x_test = get_neural_network_data("131")
    nn = Sequential(name="keras_linear_131_relu_output_activation")
    nn.add(
        Dense(
            units=3,
            input_dim=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=42
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=43
            ),
            activation="relu",
        )
    )
    nn.add(
        Dense(
            units=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=62
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=63
            ),
            activation="relu",
        )
    )
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss="mae")
    history = nn.fit(
        x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15
    )
    nn.save(this_file_dir() + "/models/keras_linear_131_relu_output_activation")

    x, y, x_test = get_neural_network_data("131")
    nn = Sequential(name="keras_linear_131_sigmoid_softplus_output_activation")
    nn.add(
        Dense(
            units=3,
            input_dim=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=42
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=43
            ),
            activation="sigmoid",
        )
    )
    nn.add(
        Dense(
            units=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=62
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=63
            ),
            activation="softplus",
        )
    )
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss="mae")
    history = nn.fit(
        x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15
    )
    nn.save(
        this_file_dir() + "/models/keras_linear_131_sigmoid_softplus_output_activation"
    )

    x, y, x_test = get_neural_network_data("131")
    nn = Sequential(name="keras_big")
    N = 100
    nn.add(
        Dense(
            units=N,
            input_dim=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=42
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=43
            ),
            activation="sigmoid",
        )
    )
    nn.add(
        Dense(
            units=N,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=42
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=43
            ),
            activation="sigmoid",
        )
    )
    nn.add(
        Dense(
            units=N,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=42
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=43
            ),
            activation="sigmoid",
        )
    )
    nn.add(
        Dense(
            units=1,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=62
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=63
            ),
            activation="softplus",
        )
    )
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss="mae")
    history = nn.fit(
        x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15
    )
    nn.save(this_file_dir() + "/models/big")

    x, y, x_test = get_neural_network_data("2353")
    nn = Sequential(name="keras_linear_2353")
    nn.add(
        Dense(
            units=3,
            input_dim=2,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=42
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=43
            ),
        )
    )
    nn.add(
        Dense(
            units=5,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=52
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=53
            ),
        )
    )
    nn.add(
        Dense(
            units=3,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=62
            ),
            bias_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=63
            ),
        )
    )
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss="mae")
    history = nn.fit(
        x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15
    )

    nn.save(this_file_dir() + "/models/keras_linear_2353")


if __name__ == "__main__":
    train_models()
