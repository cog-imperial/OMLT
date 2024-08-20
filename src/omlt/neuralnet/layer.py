r"""Neural network layer classes.

We use the following notations to define a layer:

.. math::

    \begin{align*}
        F_{in}  &:= \text{number of input features}\\
        F_{out} &:= \text{number of output features}\\
        x_i     &:= \text{the $i$-th input, $0\le i<F_{in}$}\\
        y_j     &:= \text{the $j$-th output, $0\le j<F_{out}$}\\
        w_{ij}  &:= \text{weight from $x_i$ to $y_j$, $0\le i<F_{in}, 0\le j<F_{out}$}\\
        b_j     &:= \text{bias for $y_j$, $0\le j<F_{out}$}\\
        \sigma  &:= \text{activation function}
    \end{align*}

"""

import itertools
from typing import ClassVar

import numpy as np

OUTPUT_DIMENSIONS = 3


class Layer:
    """Base layer class.

    Parameters
    ----------
    input_size : list
        size of the layer input
    output_size : list
        size of the layer output
    activation : str or None
        activation function name
    input_index_mapper : IndexMapper or None
        map indexes from this layer index to the input layer index size
    """

    def __init__(
        self, input_size, output_size, *, activation=None, input_index_mapper=None
    ):
        if not isinstance(input_size, (list, tuple)):
            msg = f"input_size must be a list or tuple, {type(input_size)} provided."
            raise TypeError(msg)
        if not isinstance(output_size, (list, tuple)):
            msg = f"output_size must be a list or tuple, {type(output_size)} provided."
            raise TypeError(msg)
        self.__input_size = list(input_size)
        self.__output_size = list(output_size)
        self.activation = activation
        if input_index_mapper is None:
            input_index_mapper = IndexMapper(input_size, input_size)
        self.__input_index_mapper = input_index_mapper

    @property
    def input_size(self):
        """Return the size of the input tensor."""
        return self.__input_size

    @property
    def output_size(self):
        """Return the size of the output tensor."""
        return self.__output_size

    @property
    def activation(self):
        """Return the activation function."""
        return self.__activation

    @activation.setter
    def activation(self, new_activation):
        """Change the activation function."""
        if new_activation is None:
            new_activation = "linear"
        self.__activation = new_activation

    @property
    def input_index_mapper(self):
        """Return the index mapper."""
        return self.__input_index_mapper

    @property
    def input_indexes_with_input_layer_indexes(self):
        """Return an iterator generating a tuple of local and input indexes.

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
        """Return a list of the input indexes."""
        return list(itertools.product(*[range(v) for v in self.__input_size]))

    @property
    def output_indexes(self):
        """Return a list of the output indexes."""
        return list(itertools.product(*[range(v) for v in self.__output_size]))

    def eval_single_layer(self, x):
        """Evaluate the layer at x.

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
        if x_reshaped.shape != tuple(self.input_size):
            msg = (
                f"Layer requires an input size {self.input_size}, but the input tensor"
                f" has size {x_reshaped.shape}."
            )
            raise ValueError(msg)
        y = self._eval(x_reshaped)
        return self._apply_activation(y)

    def __repr__(self):
        return f"<{self!s} at {hex(id(self))}>"

    def _eval(self, x):
        raise NotImplementedError

    def _apply_activation(self, x):
        if self.__activation == "linear" or self.__activation is None:
            return x
        if self.__activation == "relu":
            return np.maximum(x, 0)
        if self.__activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if self.__activation == "tanh":
            return np.tanh(x)
        msg = f"Unknown activation function {self.__activation}"
        raise ValueError(msg)


class InputLayer(Layer):
    """The first layer in any network.

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
    r"""Dense layer.

    The dense layer is defined by:

    .. math::

        \begin{align*}
            y_j = \sigma\left(\sum\limits_{i=0}^{F_{in}-1}w_{ij}x_i+b_j\right),
            && \forall 0\le j<F_{out}
        \end{align*}

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

    def __init__(  # noqa: PLR0913
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
        """Return the matrix of node weights."""
        return self.__weights

    @property
    def biases(self):
        """Return the vector of node biases."""
        return self.__biases

    def __str__(self):
        return (
            f"DenseLayer(input_size={self.input_size}, output_size={self.output_size})"
        )

    def _eval(self, x):
        y = np.dot(x, self.__weights) + self.__biases
        return np.reshape(y, tuple(self.output_size))


class GNNLayer(DenseLayer):
    r"""GNN Layer.

    We additionally introduce the following notations to describe the gnn layer:

    .. math::

        \begin{align*}
            N       &:= \text{the number of node in the graph}\\
            u       &:= \text{the node index of $x_i$, $u=\lfloor iN/F_{in}\rfloor$}\\
            v       &:= \text{the node index of $y_j$, $v=\lfloor jN/F_{out}\rfloor$}\\
            A_{u,v} &:= \text{the edge between node $u$ and $v$}\\
        \end{align*}

    The gnn layer is defined by:

    .. math::

        \begin{align*}
            y_j = \sigma \left(\sum\limits_{i=0}^{F_{in}-1}A_{u,v}w_{ij}x_i+b_j\right),
            && \forall 0\le j<F_{out},
        \end{align*}


    For example, given a GraphSAGE layer with sum aggregation:

    .. math::

        \begin{align*}
            \mathbf{y_v} =\sigma\left(\mathbf{w_1^T}\mathbf{x_v}+\mathbf{w_2}^T
            \sum\limits_{u\in\mathcal N(v)}\mathbf{x_u}+\mathbf{b}\right)
        \end{align*}

    If the graph structure is fixed, assume that it is a line graph with :math:`N=3`
    nodes, i.e., the adjacency matrix
    :math:`A=\begin{pmatrix}1 & 1 & 0\\1 & 1 & 1\\ 0 & 1 & 1\end{pmatrix}`.
    Then the corresponding GNN layer is defined with parameters:

    .. math::

        \begin{align*}
            \mathbf{W}=\begin{pmatrix}
                \mathbf{w_1} & \mathbf{w_2} & \mathbf{0} \\
                \mathbf{w_2} & \mathbf{w_1} & \mathbf{w_2} \\
                \mathbf{0} & \mathbf{w_2} & \mathbf{w_1} \\
            \end{pmatrix},
            \mathbf{B}=\begin{pmatrix}
            \mathbf{b}\\\mathbf{b}\\\mathbf{b}
            \end{pmatrix}
        \end{align*}

    Otherwise, if the input graph structure is not fixed, all weights and biases should
    be provided. In this case, the GNN layer is defined with parameters:

    .. math::

        \begin{align*}
            \mathbf{W}=\begin{pmatrix}
                \mathbf{w_1} & \mathbf{w_2} & \mathbf{w_2} \\
                \mathbf{w_2} & \mathbf{w_1} & \mathbf{w_2} \\
                \mathbf{w_2} & \mathbf{w_2} & \mathbf{w_1} \\
            \end{pmatrix},
            \mathbf{B}=\begin{pmatrix}
            \mathbf{b}\\\mathbf{b}\\\mathbf{b}
            \end{pmatrix}
        \end{align*}

    In this case, all elements :math:`A_{u,v},u\neq v` are binary variables.

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
    N : int
        number of nodes in the graph
    activation : str or None
        activation function name
    input_index_mapper : IndexMapper or None
        map indexes from this layer index to the input layer index size
    """

    def __init__(  # noqa: PLR0913
        self,
        input_size,
        output_size,
        weights,
        biases,
        N,
        *,
        activation=None,
        input_index_mapper=None,
    ):
        super().__init__(
            input_size,
            output_size,
            weights=weights,
            biases=biases,
            activation=activation,
            input_index_mapper=input_index_mapper,
        )
        if input_size[-1] % N != 0:
            msg = (
                "Input size must equal to the number of nodes multiplied by the number"
                " of input node features"
            )
            raise ValueError(msg)
        if output_size[-1] % N != 0:
            msg = (
                "Output size must equal to the number of nodes multiplied by the number"
                " of output node features"
            )
            raise ValueError(msg)
        self.__N = N
        self.__gnn_input_size = input_size[-1] // N
        self.__gnn_output_size = output_size[-1] // N

    @property
    def N(self):
        """Return the number of nodes in the graphs."""
        return self.__N

    @property
    def gnn_input_size(self):
        """Return the size of the input tensor in original GNN."""
        return self.__gnn_input_size

    @property
    def gnn_output_size(self):
        """Return the size of the output tensor in original GNN."""
        return self.__gnn_output_size

    def __str__(self):
        return f"GNNLayer(input_size={self.input_size}, output_size={self.output_size})"

    def _eval_with_adjacency(self, x, A):
        x_reshaped = (
            np.reshape(x, self.input_index_mapper.output_size)
            if self.input_index_mapper is not None
            else x[:]
        )
        y = np.zeros(shape=self.output_size)
        for output_index in self.output_indexes:
            for input_index in self.input_indexes:
                if input_index[:-1] == output_index[:-1]:
                    y[output_index] += (
                        x_reshaped[input_index]
                        * self.weights[input_index[-1], output_index[-1]]
                        * A[
                            input_index[-1] // self.gnn_input_size,
                            output_index[-1] // self.gnn_output_size,
                        ]
                    )
            y[output_index] += self.biases[output_index[-1]]

        return y


class Layer2D(Layer):
    """Abstract two-dimensional layer, downsamples values in a kernel to a single value.

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
        """Return the stride of the layer."""
        return self.__strides

    @property
    def kernel_shape(self):
        """Return the shape of the kernel."""
        raise NotImplementedError

    @property
    def kernel_depth(self):
        """Return the depth of the kernel."""
        raise NotImplementedError

    def kernel_index_with_input_indexes(self, out_d, out_r, out_c):  # noqa: ARG002
        """Kernel index with input indexes.

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

        for k_d in range(kernel_d):
            for k_r in range(kernel_r):
                for k_c in range(kernel_c):
                    input_index = (start_in_d + k_d, start_in_r + k_r, start_in_c + k_c)

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
        """Get input index.

        Returns the input index corresponding to the output at `out_index`
        and the kernel index `kernel_index`.
        """
        out_d, out_r, out_c = out_index
        for candidate_kernel_index, input_index in self.kernel_index_with_input_indexes(
            out_d, out_r, out_c
        ):
            if kernel_index == candidate_kernel_index:
                return input_index
            msg = "No input index matching the given kernel index was found."
        raise ValueError(msg)

    def _eval(self, x):
        y = np.empty(shape=self.output_size)
        if len(self.output_size) != OUTPUT_DIMENSIONS:
            msg = f"Output should have 3 dimensions but has {len(self.output_size)}"
            raise ValueError(msg)
        [depth, rows, cols] = list(self.output_size)
        for out_d in range(depth):
            for out_r in range(rows):
                for out_c in range(cols):
                    y[out_d, out_r, out_c] = self._eval_at_index(x, out_d, out_r, out_c)
        return y

    def _eval_at_index(self, x, out_d, out_r, out_c):
        raise NotImplementedError


class PoolingLayer2D(Layer2D):
    """Two-dimensional pooling layer.

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

    _POOL_FUNCTIONS: ClassVar = {"max": max}

    def __init__(  # noqa: PLR0913
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
        if pool_func_name not in PoolingLayer2D._POOL_FUNCTIONS:
            msg = (
                f"Allowable pool functions are {PoolingLayer2D._POOL_FUNCTIONS},"
                f" {pool_func_name} was provided."
            )
            raise ValueError(msg)
        self._pool_func_name = pool_func_name
        self._kernel_shape = kernel_shape
        self._kernel_depth = kernel_depth

    @property
    def kernel_shape(self):
        """Return the shape of the kernel."""
        return self._kernel_shape

    @property
    def kernel_depth(self):
        """Return the depth of the kernel."""
        return self._kernel_depth

    def __str__(self):
        return (
            f"PoolingLayer(input_size={self.input_size}, output_size={self.output_size}"
            f", strides={self.strides}, kernel_shape={self.kernel_shape}),"
            f" pool_func_name={self._pool_func_name}"
        )

    def _eval_at_index(self, x, out_d, out_r, out_c):
        vals = [
            x[index]
            for (_, index) in self.kernel_index_with_input_indexes(out_d, out_r, out_c)
        ]
        pool_func = PoolingLayer2D._POOL_FUNCTIONS[self._pool_func_name]
        return pool_func(vals)


class ConvLayer2D(Layer2D):
    """Two-dimensional convolutional layer.

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

    def __init__(  # noqa: PLR0913
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
        """Kernel with input indexes.

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
        """Return the shape of the cross-correlation kernel."""
        return self.__kernel.shape[2:]

    @property
    def kernel_depth(self):
        """Return the depth of the cross-correlation kernel."""
        return self.__kernel.shape[1]

    @property
    def kernel(self):
        """Return the cross-correlation kernel."""
        return self.__kernel

    def __str__(self):
        return (
            f"ConvLayer(input_size={self.input_size}, output_size={self.output_size},"
            f" strides={self.strides}, kernel_shape={self.kernel_shape})"
        )

    def _eval_at_index(self, x, out_d, out_r, out_c):
        acc = 0.0
        for k, index in self.kernel_with_input_indexes(out_d, out_r, out_c):
            acc += k * x[index]
        return acc


class IndexMapper:
    """Map indexes from one layer to the other.

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
        """Return the size of the input tensor."""
        return self.__input_size

    @property
    def output_size(self):
        """Return the size of the output tensor."""
        return self.__output_size

    def __call__(self, index):
        flat_index = np.ravel_multi_index(index, self.__output_size)
        return np.unravel_index(flat_index, self.__input_size)

    def __str__(self):
        return (
            f"IndexMapper(input_size={self.input_size}, output_size={self.output_size})"
        )
