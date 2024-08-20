"""Scaling.

The omlt.scaling module describes the interface for providing different scaling
expressions to the Pyomo model for the inputs and outputs of an ML model. An
implementation of a common scaling approach is included with `OffsetScaling`.
"""

import abc
from typing import Any


class ScalingInterface(abc.ABC):
    @abc.abstractmethod
    def get_scaled_input_expressions(self, input_vars):
        """Get scaled inputs.

        This method returns a list of expressions for the scaled inputs from
        the unscaled inputs
        """
        # pragma: no cover

    @abc.abstractmethod
    def get_unscaled_output_expressions(self, scaled_output_vars):
        """Get unscaled outputs.

        This method returns a list of expressions for the unscaled outputs from
        the scaled outputs
        """
        # pragma: no cover


def convert_to_dict(x: Any) -> dict[Any, Any]:
    if isinstance(x, dict):
        return dict(x)
    return dict(enumerate(x))


class OffsetScaling(ScalingInterface):
    r"""OffsetScaling interface.

    This scaling object represents the following scaling equations for inputs (x)
    and outputs (y)

    .. math::
        \begin{align*}
        x_i^{scaled} = \frac{(x_i-x_i^{offset})}{x_i^{factor}} \\
        y_i^{scaled} = \frac{(y_i-y_i^{offset})}{y_i^{factor}}
        \end{align*}

    Parameters
    ----------
    offset_inputs : array-like
        Array of the values of the offsets for each input to the network
    factor_inputs : array-like
        Array of the scaling factors (division) for each input to the network
    offset_outputs : array-like
        Array of the values of the offsets for each output from the network
    factor_outputs : array-like
        Array of the scaling factors (division) for each output from the network
    """

    def __init__(self, offset_inputs, factor_inputs, offset_outputs, factor_outputs):
        super().__init__()
        self.__x_offset = convert_to_dict(offset_inputs)
        self.__x_factor = convert_to_dict(factor_inputs)
        self.__y_offset = convert_to_dict(offset_outputs)
        self.__y_factor = convert_to_dict(factor_outputs)

        for k, v in self.__x_factor.items():
            if v <= 0:
                msg = (
                    "OffsetScaling only accepts positive values"
                    " for factor_inputs. Negative value found at"
                    f" index {k}."
                )
                raise ValueError(msg)
        for k, v in self.__y_factor.items():
            if v <= 0:
                msg = (
                    "OffsetScaling only accepts positive values"
                    " for factor_outputs. Negative value found at"
                    f" index {k}."
                )
                raise ValueError(msg)

    def get_scaled_input_expressions(self, input_vars):
        """Get the scaled input expressions of the input variables."""
        sorted_keys = sorted(input_vars.keys())
        if (
            sorted(self.__x_offset) != sorted_keys
            or sorted(self.__x_factor) != sorted_keys
        ):
            msg = (
                "get_scaled_input_expressions called with input_vars"
                " that do not have the same indices as offset_inputs"
                " or factor_inputs.\n"
                f"Keys in input_vars: {sorted_keys}.\n"
                f"Keys in offset_inputs: {sorted(self.__x_offset)}.\n"
                f"Keys in offset_factor: {sorted(self.__x_factor)}."
            )
            raise ValueError(msg)
        x = input_vars
        return {k: (x[k] - self.__x_offset[k]) / self.__x_factor[k] for k in x}

    def get_unscaled_input_expressions(self, scaled_input_vars):
        """Get the unscaled input expressions of the scaled input variables."""
        sorted_keys = sorted(scaled_input_vars.keys())
        if (
            sorted(self.__x_offset) != sorted_keys
            or sorted(self.__x_factor) != sorted_keys
        ):
            msg = (
                "get_scaled_input_expressions called with input_vars"
                " that do not have the same indices as offset_inputs"
                " or factor_inputs.\n"
                f"Keys in input_vars: {sorted_keys}\n"
                f"Keys in offset_inputs: {sorted(self.__x_offset)}\n"
                f"Keys in offset_factor: {sorted(self.__x_factor)}"
            )
            raise ValueError(msg)

        scaled_x = scaled_input_vars
        return {
            k: scaled_x[k] * self.__x_factor[k] + self.__x_offset[k] for k in scaled_x
        }

    def get_scaled_output_expressions(self, output_vars):
        """Get the scaled output expressions of the output variables."""
        sorted_keys = sorted(output_vars.keys())
        if (
            sorted(self.__y_offset) != sorted_keys
            or sorted(self.__y_factor) != sorted_keys
        ):
            msg = (
                "get_scaled_output_expressions called with output_vars"
                " that do not have the same indices as offset_outputs"
                " or factor_outputs.\n"
                f"Keys in output_vars: {sorted_keys}\n"
                f"Keys in offset_outputs: {sorted(self.__y_offset)}\n"
                f"Keys in offset_factor: {sorted(self.__y_factor)}"
            )
            raise ValueError(msg)

        y = output_vars
        return {k: (y[k] - self.__y_offset[k]) / self.__y_factor[k] for k in y}

    def get_unscaled_output_expressions(self, scaled_output_vars):
        """Get the unscaled output expressions of the scaled output variables."""
        sorted_keys = sorted(scaled_output_vars.keys())
        if (
            sorted(self.__y_offset) != sorted_keys
            or sorted(self.__y_factor) != sorted_keys
        ):
            msg = (
                "get_scaled_output_expressions called with output_vars"
                " that do not have the same indices as offset_outputs"
                " or factor_outputs.\n"
                f"Keys in output_vars: {sorted_keys}\n"
                f"Keys in offset_outputs: {sorted(self.__y_offset)}\n"
                f"Keys in offset_factor: {sorted(self.__y_factor)}"
            )
            raise ValueError(msg)

        scaled_y = scaled_output_vars
        return {
            k: scaled_y[k] * self.__y_factor[k] + self.__y_offset[k] for k in scaled_y
        }
