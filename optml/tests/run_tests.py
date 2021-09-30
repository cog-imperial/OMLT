from optml.tests.test_input_output import *
from optml.tests.test_keras import *
from optml.tests.test_keras_reader import *
from optml.tests.test_network_definition import *
from optml.tests.test_neural_net import *
from optml.tests.test_relu import *
from optml.tests.test_scaling import *

if __name__ == '__main__':
    test_input_output_auto_creation()
    test_provided_inputs_outputs()
    test_keras_linear_131_full()
    test_keras_linear_big()
    test_keras_reader()
    test_network_definition()
    xtest_two_node_pass_variables()
    test_two_node_reduced_space_1()
    test_two_node_full_space()
    test_two_node_bigm()
    test_two_node_complementarity()
    test_offset_scaling()
