"""Neural network layer classes."""
import itertools
import numpy as np

class Layer:
    """
    Base layer class.
    
    Parameters
    ----------
    input_size : tuple
        size of the layer input
    output_size : tuple
        size of the layer output
    activation : str or None
        activation function name
    input_index_mapper : IndexMapper or None
        map indexes from this layer index to the input layer index size
    """
    def __init__(self, input_size, output_size, *, activation=None, input_index_mapper=None):
        assert isinstance(input_size, list)
        assert isinstance(output_size, list)
        if activation is None:
            activation = "linear"
        self.__input_size = input_size
        self.__output_size = output_size
        self.__activation = activation
        self.__input_index_mapper = input_index_mapper

    @property
    def input_size(self):
        """Return the size of the input tensor"""
        return self.__input_size

    @property
    def output_size(self):
        """Return the size of the output tensor"""
        return self.__output_size

    @property
    def activation(self):
        """Return the activation function"""
        return self.__activation

    @property
    def input_index_mapper(self):
        """Return the index mapper"""
        return self.__input_index_mapper

    @property
    def input_indexes_with_input_layer_indexes(self):
        """
        Return an iterator generating a tuple of local and input indexes.

        Local indexes are indexes over the elements of the current layer.
        Input indexes are indexes over the elements of the previous layer.
        """
        if self.__input_index_mapper is None:
            for index in self.input_indexes:
                yield index, index
        else:
            mapper = self.__input_index_mapper
            for index in self.input_indexes:
                yield index, mapper(index)

    @property
    def input_indexes(self):
        """Return a list of the input indexes"""
        return list(itertools.product(*[range(v) for v in self.__input_size]))

    @property
    def output_indexes(self):
        """Return a list of the output indexes"""
        return list(itertools.product(*[range(v) for v in self.__output_size]))

    def eval(self, x):
        """
        Evaluate the layer at x.
        
        Parameters
        ----------
        x : array-like
            the input tensor. Must have size `self.input_size`.
        """
        if self.__input_index_mapper is not None:
            x = np.reshape(x, self.__input_index_mapper.output_size)
        assert x.shape == tuple(self.input_size)
        y = self._eval(x)
        return self._apply_activation(y)

    def __repr__(self):
        return f"<{str(self)} at {hex(id(self))}>"

    def _eval(self, x):
        raise NotImplementedError()

    def _apply_activation(self, x):
        if self.__activation == "linear" or self.__activation is None:
            return x
        elif self.__activation == "relu":
            return np.maximum(x, 0)
        elif self.__activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        elif self.__activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unknown activation function {self.__activation}")


class InputLayer(Layer):
    """
    The first layer in any network.

    Parameters
    ----------
    size : tuple
        the size of the input.
    """
    def __init__(self, size):
        super().__init__(size, size)

    def __str__(self):
        return f"InputLayer(input_size={self.input_size}, output_size={self.output_size})"

    def _eval(self, x):
        return x


class DenseLayer(Layer):
    """
    Dense layer implementing `output = activation(dot(input, weights) + biases)`.

    Parameters
    ----------
    input_size : tuple
        the size of the input.
    output_size : tuple
        the size of the output.
    weight : matrix-like
        the weight matrix.
    biases : array-like
        the biases.
    activation : str or None
        activation function name
    input_index_mapper : IndexMapper or None
        map indexes from this layer index to the input layer index size
    """
    def __init__(self, input_size, output_size, weights, biases, *, activation=None, input_index_mapper=None):
        super().__init__(input_size, output_size, activation=activation, input_index_mapper=input_index_mapper)
        self.__weights = weights
        self.__biases = biases

    @property
    def weights(self):
        return self.__weights

    @property
    def biases(self):
        return self.__biases

    def __str__(self):
        return f"DenseLayer(input_size={self.input_size}, output_size={self.output_size})"

    def _eval(self, x):
        y = np.dot(x, self.__weights) + self.__biases
        y = np.reshape(y, tuple(self.output_size))
        assert y.shape == tuple(self.output_size)
        return y


class ConvLayer(Layer):
    """
    Two-dimensional convolutional layer.

    Parameters
    ----------
    input_size : tuple
        the size of the input.
    output_size : tuple
        the size of the output.
    strides : matrix-like
        stride of the cross-correlation kernel.
    kernel : matrix-like
        the cross-correlation kernel.
    activation : str or None
        activation function name
    input_index_mapper : IndexMapper or None
        map indexes from this layer index to the input layer index size
    """
    def __init__(self, input_size, output_size, strides, kernel, *, activation=None, input_index_mapper=None):
        super().__init__(input_size, output_size, activation=activation, input_index_mapper=input_index_mapper)
        self.__strides = strides
        self.__kernel = kernel

    def kernel_with_input_indexes(self, out_d, out_r, out_c):
        """
        Returns an iterator over the kernel value and input index 
        for the output at index `(out_d, out_r, out_c)`.

        Parameters
        ----------
        out_d : int
            the output depth.
        out_d : int
            the output row.
        out_c : int
            the output column.
        """
        [_, kernel_d, kernel_r, kernel_c] = self.__kernel.shape
        [rows_stride, cols_stride] = self.__strides
        start_in_d = 0
        start_in_r = out_r * rows_stride
        start_in_c = out_c * cols_stride
        mapper = lambda x: x
        if self.input_index_mapper is not None:
            mapper = self.input_index_mapper

        for k_d in range(kernel_d):
            for k_r in range(kernel_r):
                for k_c in range(kernel_c):
                    k_v = self.__kernel[out_d, k_d, k_r, k_c]
                    local_index = (start_in_d + k_d, start_in_r + k_r, start_in_c + k_c)
                    yield k_v, mapper(local_index)

    @property
    def strides(self):
        return self.__strides

    @property
    def kernel_shape(self):
        return self.__kernel.shape[2:]

    @property
    def kernel(self):
        return self.__kernel

    def __str__(self):
        return f"ConvLayer(input_size={self.input_size}, output_size={self.output_size}, strides={self.strides}, kernel_shape={self.kernel_shape})"

    def _eval(self, x):
        y = np.empty(shape=self.output_size)
        assert len(self.output_size) == 3
        [depth, rows, cols] = self.output_size
        for out_d in range(depth):
            for out_r in range(rows):
                for out_c in range(cols):
                    acc = 0.0
                    for (k, index) in self.kernel_with_input_indexes(out_d, out_r, out_c):
                        acc += k * x[index]
                    y[out_d, out_r, out_c] = acc
        return y


class IndexMapper:
    """
    Map indexes from one layer to the other.

    Parameters
    ----------
    input_size : tuple
        the input size
    output_size : tuple
        the mapped input layer's output size
    """
    def __init__(self, input_size, output_size):
        self.__input_size = input_size
        self.__output_size = output_size

    @property
    def input_size(self):
        return self.__input_size

    @property
    def output_size(self):
        return self.__output_size

    def __call__(self, index):
        flat_index = np.ravel_multi_index(index, self.__output_size)
        return np.unravel_index(flat_index, self.__input_size)

    def __str__(self):
        return f"IndexMapper(input_size={self.input_size}, output_size={self.output_size})"
