"""
The omlt.scaling module describes the interface for providing different scaling
expressions to the Pyomo model for the inputs and outputs of an ML model. An implementation of a common scaling approach is
included with `OffsetScaling`.
"""

import abc


class ScalingInterface(abc.ABC):
    @abc.abstractmethod
    def get_scaled_input_expressions(self, input_vars):
        """This method returns a list of expressions for the scaled inputs from
        the unscaled inputs"""
        pass  # pragma: no cover

    @abc.abstractmethod
    def get_unscaled_output_expressions(self, scaled_output_vars):
        """This method returns a list of expressions for the unscaled outputs from
        the scaled outputs"""
        pass  # pragma: no cover


def convert_to_dict(x):
    if type(x) is dict:
        return dict(x)
    return {i: v for i, v in enumerate(x)}


class OffsetScaling(ScalingInterface):
    r"""
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

        super(OffsetScaling, self).__init__()
        self.__x_offset = convert_to_dict(offset_inputs)
        self.__x_factor = convert_to_dict(factor_inputs)
        self.__y_offset = convert_to_dict(offset_outputs)
        self.__y_factor = convert_to_dict(factor_outputs)

        for k, v in self.__x_factor.items():
            if v <= 0:
                raise ValueError(
                    "OffsetScaling only accepts positive values"
                    " for factor_inputs. Negative value found at"
                    " index {}.".format(k)
                )
        for k, v in self.__y_factor.items():
            if v <= 0:
                raise ValueError(
                    "OffsetScaling only accepts positive values"
                    " for factor_outputs. Negative value found at"
                    " index {}.".format(k)
                )

    def get_scaled_input_expressions(self, input_vars):
        """
        Get the scaled input expressions of the input variables.
        """
        sorted_keys = sorted(input_vars.keys())
        if (
            sorted(self.__x_offset) != sorted_keys
            or sorted(self.__x_factor) != sorted_keys
        ):
            raise ValueError(
                "get_scaled_input_expressions called with input_vars"
                " that do not have the same indices as offset_inputs"
                " or factor_inputs.\n"
                "Keys in input_vars: {}.\n"
                "Keys in offset_inputs: {}.\n"
                "Keys in offset_factor: {}.".format(
                    sorted_keys, sorted(self.__x_offset), sorted(self.__x_factor)
                )
            )
        x = input_vars
        return {k: (x[k] - self.__x_offset[k]) / self.__x_factor[k] for k in x.keys()}

    def get_unscaled_input_expressions(self, scaled_input_vars):
        """
        Get the unscaled input expressions of the scaled input variables.
        """
        sorted_keys = sorted(scaled_input_vars.keys())
        if (
            sorted(self.__x_offset) != sorted_keys
            or sorted(self.__x_factor) != sorted_keys
        ):
            raise ValueError(
                "get_scaled_input_expressions called with input_vars"
                " that do not have the same indices as offset_inputs"
                " or factor_inputs.\n"
                "Keys in input_vars: {}\n"
                "Keys in offset_inputs: {}\n"
                "Keys in offset_factor: {}".format(
                    sorted_keys, sorted(self.__x_offset), sorted(self.__x_factor)
                )
            )

        scaled_x = scaled_input_vars
        return {
            k: scaled_x[k] * self.__x_factor[k] + self.__x_offset[k]
            for k in scaled_x.keys()
        }

    def get_scaled_output_expressions(self, output_vars):
        """
        Get the scaled output expressions of the output variables.
        """
        sorted_keys = sorted(output_vars.keys())
        if (
            sorted(self.__y_offset) != sorted_keys
            or sorted(self.__y_factor) != sorted_keys
        ):
            raise ValueError(
                "get_scaled_output_expressions called with output_vars"
                " that do not have the same indices as offset_outputs"
                " or factor_outputs.\n"
                "Keys in output_vars: {}\n"
                "Keys in offset_outputs: {}\n"
                "Keys in offset_factor: {}".format(
                    sorted_keys, sorted(self.__y_offset), sorted(self.__y_factor)
                )
            )

        y = output_vars
        return {k: (y[k] - self.__y_offset[k]) / self.__y_factor[k] for k in y.keys()}

    def get_unscaled_output_expressions(self, scaled_output_vars):
        """
        Get the unscaled output expressions of the scaled output variables.
        """
        sorted_keys = sorted(scaled_output_vars.keys())
        if (
            sorted(self.__y_offset) != sorted_keys
            or sorted(self.__y_factor) != sorted_keys
        ):
            raise ValueError(
                "get_scaled_output_expressions called with output_vars"
                " that do not have the same indices as offset_outputs"
                " or factor_outputs.\n"
                "Keys in output_vars: {}\n"
                "Keys in offset_outputs: {}\n"
                "Keys in offset_factor: {}".format(
                    sorted_keys, sorted(self.__y_offset), sorted(self.__y_factor)
                )
            )

        scaled_y = scaled_output_vars
        return {
            k: scaled_y[k] * self.__y_factor[k] + self.__y_offset[k]
            for k in scaled_y.keys()
        }
