import keras
from conftest import get_neural_network_data
from keras.layers import Conv2D, Dense
from keras.models import Sequential
from keras.optimizers import Adamax
from pyomo.common.fileutils import this_file_dir

from omlt.io import write_onnx_model_with_bounds


def train_models():  # noqa: PLR0915
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
    nn.fit(x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15)
    nn.save(this_file_dir() + "/models/keras_linear_131.keras")

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
    nn.fit(x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15)
    nn.save(this_file_dir() + "/models/keras_linear_131_sigmoid.keras")

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
    nn.fit(x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15)
    nn.save(
        this_file_dir() + "/models/keras_linear_131_sigmoid_output_activation.keras"
    )

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
    nn.fit(x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15)
    nn.save(this_file_dir() + "/models/keras_linear_131_relu.keras")

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
    nn.fit(x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15)
    nn.save(this_file_dir() + "/models/keras_linear_131_relu_output_activation.keras")

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
    nn.fit(x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15)
    nn.save(
        this_file_dir()
        + "/models/keras_linear_131_sigmoid_softplus_output_activation.keras"
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
    nn.fit(x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15)
    nn.save(this_file_dir() + "/models/big.keras")

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
    nn.fit(x=x, y=y, validation_split=0.2, batch_size=16, verbose=1, epochs=15)

    nn.save(this_file_dir() + "/models/keras_linear_2353.keras")


def train_conv():
    nn = Sequential(name="keras_conv_7x7_relu")
    nn.add(
        Conv2D(
            filters=1,
            kernel_size=(2, 2),
            activation="relu",
            data_format="channels_first",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=1.0, stddev=0.05, seed=62
            ),
            input_shape=(1, 7, 7),
        )
    )
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss="mae")
    import tempfile

    import tf2onnx

    onnx_model, _ = tf2onnx.convert.from_keras(nn)

    input_bounds = {}
    for i in range(7):
        for j in range(7):
            input_bounds[0, i, j] = (0.0, 1.0)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        write_onnx_model_with_bounds(f.name, onnx_model, input_bounds)
        print(f"Wrote ONNX model with bounds at {f.name}")  # noqa: T201


if __name__ == "__main__":
    train_models()
    train_conv()
