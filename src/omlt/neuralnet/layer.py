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

    def __init__(
        self, input_size, output_size, *, activation=None, input_index_mapper=None
    ):
        assert isinstance(input_size, list)
        assert isinstance(output_size, list)
        self.__input_size = input_size
        self.__output_size = output_size
        self.activation = activation
        if input_index_mapper is None:
            input_index_mapper = IndexMapper(input_size, input_size)
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

    @activation.setter
    def activation(self, new_activation):
        """Change the activation function"""
        if new_activation is None:
            new_activation = "linear"
        self.__activation = new_activation

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

    def eval_single_layer(self, x):
        """
        Evaluate the layer at x.

        Parameters
        ----------
        x : array-like
            the input tensor. Must have size `self.input_size`.
        """
        x_reshaped = (
            np.reshape(x, self.__input_index_mapper.output_size)
            if self.__input_index_mapper is not None
            else x[:]
        )
        assert x_reshaped.shape == tuple(self.input_size)
        y = self._eval(x_reshaped)
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
        elif self.__activation == "tanh":
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
        return (
            f"InputLayer(input_size={self.input_size}, output_size={self.output_size})"
        )

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

    def __init__(
        self,
        input_size,
        output_size,
        weights,
        biases,
        *,
        activation=None,
        input_index_mapper=None,
    ):
        super().__init__(
            input_size,
            output_size,
            activation=activation,
            input_index_mapper=input_index_mapper,
        )
        self.__weights = weights
        self.__biases = biases

    @property
    def weights(self):
        """Return the matrix of node weights"""
        return self.__weights

    @property
    def biases(self):
        """Return the vector of node biases"""
        return self.__biases

    def __str__(self):
        return (
            f"DenseLayer(input_size={self.input_size}, output_size={self.output_size})"
        )

    def _eval(self, x):
        y = np.dot(x, self.__weights) + self.__biases
        y = np.reshape(y, tuple(self.output_size))
        assert y.shape == tuple(self.output_size)
        return y


class Layer2D(Layer):
    """
    Abstract two-dimensional layer that downsamples values in a kernel to a single value.

    Parameters
    ----------
    input_size : tuple
        the size of the input.
    output_size : tuple
        the size of the output.
    strides : matrix-like
        stride of the kernel.
    activation : str or None
        activation function name
    input_index_mapper : IndexMapper or None
        map indexes from this layer index to the input layer index size
    """

    def __init__(
        self,
        input_size,
        output_size,
        strides,
        *,
        activation=None,
        input_index_mapper=None,
    ):
        super().__init__(
            input_size,
            output_size,
            activation=activation,
            input_index_mapper=input_index_mapper,
        )
        self.__strides = strides

    @property
    def strides(self):
        """Return the stride of the layer"""
        return self.__strides

    @property
    def kernel_shape(self):
        """Return the shape of the kernel"""
        raise NotImplementedError()

    @property
    def kernel_depth(self):
        """Return the depth of the kernel"""
        raise NotImplementedError()

    def kernel_index_with_input_indexes(self, out_d, out_r, out_c):
        """
        Returns an iterator over the index within the kernel and input index
        for the output at index `(out_d, out_r, out_c)`.

        Parameters
        ----------
        out_d : int
            the output depth.
        out_r : int
            the output row.
        out_c : int
            the output column.
        """
        kernel_d = self.kernel_depth
        [kernel_r, kernel_c] = self.kernel_shape
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
                    input_index = (start_in_d + k_d, start_in_r + k_r, start_in_c + k_c)
                    assert len(input_index) == len(self.input_size)
                    # don't yield an out-of-bounds input index;
                    # can happen if ceil mode is enabled for pooling layers
                    # as this could require using a partial kernel
                    # even though we loop over ALL kernel indexes.
                    if not all(
                        input_index[i] < self.input_size[i]
                        for i in range(len(input_index))
                    ):
                        continue
                    yield (k_d, k_r, k_c), input_index

    def get_input_index(self, out_index, kernel_index):
        """
        Returns the input index corresponding to the output at `out_index`
        and the kernel index `kernel_index`.
        """
        out_d, out_r, out_c = out_index
        for candidate_kernel_index, input_index in self.kernel_index_with_input_indexes(
            out_d, out_r, out_c
        ):
            if kernel_index == candidate_kernel_index:
                return input_index

    def _eval(self, x):
        y = np.empty(shape=self.output_size)
        assert len(self.output_size) == 3
        [depth, rows, cols] = self.output_size
        for out_d in range(depth):
            for out_r in range(rows):
                for out_c in range(cols):
                    y[out_d, out_r, out_c] = self._eval_at_index(x, out_d, out_r, out_c)
        return y

    def _eval_at_index(self, x, out_d, out_r, out_c):
        raise NotImplementedError()


class PoolingLayer2D(Layer2D):
    """
    Two-dimensional pooling layer.

    Parameters
    ----------
    input_size : tuple
        the size of the input.
    output_size : tuple
        the size of the output.
    strides : matrix-like
        stride of the kernel.
    pool_func : str
        name of function used to pool values in a kernel to a single value.
    transpose : bool
        True iff input matrix is accepted in transposed (i.e. column-major)
        form.
    activation : str or None
        activation function name
    input_index_mapper : IndexMapper or None
        map indexes from this layer index to the input layer index size
    """

    _POOL_FUNCTIONS = {"max": max}

    def __init__(
        self,
        input_size,
        output_size,
        strides,
        pool_func_name,
        kernel_shape,
        kernel_depth,
        *,
        activation=None,
        input_index_mapper=None,
    ):
        super().__init__(
            input_size,
            output_size,
            strides,
            activation=activation,
            input_index_mapper=input_index_mapper,
        )
        self._pool_func_name = pool_func_name
        self._kernel_shape = kernel_shape
        self._kernel_depth = kernel_depth

    @property
    def kernel_shape(self):
        """Return the shape of the kernel"""
        return self._kernel_shape

    @property
    def kernel_depth(self):
        """Return the depth of the kernel"""
        return self._kernel_depth

    def __str__(self):
        return f"PoolingLayer(input_size={self.input_size}, output_size={self.output_size}, strides={self.strides}, kernel_shape={self.kernel_shape}), pool_func_name={self._pool_func_name}"

    def _eval_at_index(self, x, out_d, out_r, out_c):
        vals = [
            x[index]
            for (_, index) in self.kernel_index_with_input_indexes(out_d, out_r, out_c)
        ]
        assert self._pool_func_name in PoolingLayer2D._POOL_FUNCTIONS
        pool_func = PoolingLayer2D._POOL_FUNCTIONS[self._pool_func_name]
        return pool_func(vals)


class ConvLayer2D(Layer2D):
    """
    Two-dimensional convolutional layer.

    Parameters
    ----------
    input_size : tuple
        the size of the input.
    output_size : tuple
        the size of the output..
    strides : matrix-like
        stride of the cross-correlation kernel.
    kernel : matrix-like
        the cross-correlation kernel.
    activation : str or None
        activation function name
    input_index_mapper : IndexMapper or None
        map indexes from this layer index to the input layer index size
    """

    def __init__(
        self,
        input_size,
        output_size,
        strides,
        kernel,
        *,
        activation=None,
        input_index_mapper=None,
    ):
        super().__init__(
            input_size,
            output_size,
            strides,
            activation=activation,
            input_index_mapper=input_index_mapper,
        )
        self.__kernel = kernel

    def kernel_with_input_indexes(self, out_d, out_r, out_c):
        """
        Returns an iterator over the kernel value and input index
        for the output at index `(out_d, out_r, out_c)`.

        Parameters
        ----------
        out_d : int
            the output depth.
        out_r : int
            the output row.
        out_c : int
            the output column.
        """
        for (k_d, k_r, k_c), input_index in self.kernel_index_with_input_indexes(
            out_d, out_r, out_c
        ):
            k_v = self.__kernel[out_d, k_d, k_r, k_c]
            yield k_v, input_index

    @property
    def kernel_shape(self):
        """Return the shape of the cross-correlation kernel"""
        return self.__kernel.shape[2:]

    @property
    def kernel_depth(self):
        """Return the depth of the cross-correlation kernel"""
        return self.__kernel.shape[1]

    @property
    def kernel(self):
        """Return the cross-correlation kernel"""
        return self.__kernel

    def __str__(self):
        return f"ConvLayer(input_size={self.input_size}, output_size={self.output_size}, strides={self.strides}, kernel_shape={self.kernel_shape})"

    def _eval_at_index(self, x, out_d, out_r, out_c):
        acc = 0.0
        for (k, index) in self.kernel_with_input_indexes(out_d, out_r, out_c):
            acc += k * x[index]
        return acc


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
        """Return the size of the input tensor"""
        return self.__input_size

    @property
    def output_size(self):
        """Return the size of the output tensor"""
        return self.__output_size

    def __call__(self, index):
        flat_index = np.ravel_multi_index(index, self.__output_size)
        return np.unravel_index(flat_index, self.__input_size)

    def __str__(self):
        return (
            f"IndexMapper(input_size={self.input_size}, output_size={self.output_size})"
        )
