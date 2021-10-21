import abc

"""
This module describes the interface for providing different scaling 
expressions to the Pyomo model for the inputs and outputs of the
neural network. An implementation of a common scaling approach is 
included below.
"""


class ScalingInterface(abc.ABC):
    @abc.abstractmethod
    def get_scaled_input_expressions(self, input_vars):
        """This method returns a list of expressions for the scaled inputs from
        the unscaled inputs"""
        pass

    @abc.abstractmethod
    def get_unscaled_output_expressions(self, scaled_output_vars):
        """This method returns a list of expressions for the unscaled outputs from
        the scaled outputs"""
        pass


class OffsetScaling(ScalingInterface):
    def __init__(self, offset_inputs, factor_inputs, offset_outputs, factor_outputs):
        """
        This scaling object represents the following scaling equations for inputs (x)
        and outputs (y)

        scaled_x_i = (x_i-offset_inputs_i)/factor_inputs_i
        scaled_y_i = (y_i-offset_outputs_i)/factor_outputs_i

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
        super(OffsetScaling, self).__init__()
        self.__x_offset = offset_inputs
        self.__x_factor = factor_inputs
        self.__y_offset = offset_outputs
        self.__y_factor = factor_outputs

    def get_scaled_input_expressions(self, input_vars):
        x = input_vars
        if isinstance(x, dict):
            ret = {}
            for i, j in enumerate(x.keys()):
                ret[j] = (x[j] - self.__x_offset[i]) / self.__x_factor[i]
            return ret
        else:
            return [
                (x[i] - self.__x_offset[i]) / self.__x_factor[i] for i in range(len(x))
            ]

    def get_scaled_output_expressions(self, output_vars):
        y = output_vars
        if isinstance(y, dict):
            ret = {}
            for i, j in enumerate(y.keys()):
                ret[j] = (y[j] - self.__y_offset[i]) / self.__y_factor[i]
            return ret
        else:
            return [
                (y[i] - self.__y_offset[i]) / self.__y_factor[i] for i in range(len(y))
            ]

    def get_unscaled_output_expressions(self, scaled_output_vars):
        scaled_y = scaled_output_vars
        if isinstance(scaled_y, dict):
            ret = {}
            for i, j in enumerate(scaled_y.keys()):
                ret[j] = scaled_y[j] * self.__y_factor[i] + self.__y_offset[i]
            return ret
        else:
            return [
                scaled_y[i] * self.__y_factor[i] + self.__y_offset[i]
                for i in range(len(scaled_y))
            ]
