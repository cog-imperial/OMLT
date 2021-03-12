import pyomo.environ as pyo
from pyomo.core.base.block import _BlockData, declare_custom_block
from .utils import _sequential_to_dict

class EncodedMode:

    def build(self,block):
        """
            Build a pyomo model that encodes a neural network into multiple variables and constraints. Each activation function is encoded with pre-activation and post-activation variables.
        """
        m = block
        self.block = block
        n_inputs = self.block.n_inputs
        n_outputs = self.block.n_outputs
        n_nodes = self.block.n_nodes

        m.INPUTS = pyo.Set(initialize=list(range(n_inputs)), ordered=True)
        m.NODES = pyo.Set(initialize=list(range(n_nodes)), ordered=True)
        m.INTERMEDIATE_NODES = pyo.Set(initialize=list(range(n_inputs, n_nodes)), ordered=True)
        m.OUTPUTS = pyo.Set(initialize=list(range(n_nodes, n_nodes+n_outputs)), ordered=True)

        #input variables
        m.x = pyo.Var(m.INPUTS)

        # pre-activation values
        m.zhat = pyo.Var(m.INTERMEDIATE_NODES)

        # post-activation values
        m.z = pyo.Var(m.NODES)

        # output variables
        m.y = pyo.Var(m.OUTPUTS)

        @m.Constraint(m.INPUTS)
        def _inputs(m,i):
            return m.z[i] == m.x[i]

        @m.Constraint(m.INTERMEDIATE_NODES)
        def _linear(m,i):
            return m.zhat[i] == sum(m.w[i][j]*m.z[j] for j in m.w[i]) + m.b[i]

        @m.Constraint(m.OUTPUTS)
        def _outputs(m,i):
            return m.y[i] == sum(m.w[i][j]*m.z[j] for j in m.w[i]) + m.b[i]

        #Activation constraints
        @m.Constraint(m.INTERMEDIATE_NODES)
        def _z_activation(m,i):
            return m.z[i] == m.activation(m.zhat[i])

#TODO: build up a dense pyomo constraint
class SingleConstraintMode:
    def build(self,block):
        """
            Build a pyomo model that encodes a neural network as a pyomo constraint
        """
        m = block
        self.block = block
        n_inputs = m.n_inputs
        n_outputs = m.n_outputs
        n_nodes = m.n_nodes

        m.x = pyo.Var(m.INPUTS)
        m.y = pyo.Var(m.OUTPUTS)

        @m.Constraint()
        def _neural_constraint(m):

            return 0

#Entire neural net can be one dense constraint OR neural net can be encoded into layered logic
@declare_custom_block(name='NeuralBlock')
class NeuralBlockData(_BlockData):

    def set_neural_data(self,n_inputs,n_outputs,n_nodes,w,b,activation = pyo.tanh):
        """
        This functions sets up a Pyomo model to capture a ReLU neural network.

        Parameters
        ----------
           n_inputs : int
              The number of inputs to the network
           n_outputs : int
              The number of outputs from the network
           n_nodes : int
              The total number of nodes in the network (equal to n_inputs + # of intermediate nodes)
           w : dict of dict of floats
              The weights for the network. This is a sparse structure indexed by w[i][j] where
              w[i] is a dictionary of dictionaries, and w[i][j] is a dictionary of floats.
              All indices are integer, corresponding to the node numbers 0 -> n_nodes
           b : dict of floats
              The biases for the network. This is a dictionary of floats and should have length of
              n_nodes - n_inputs.
           activation : pyomo function
              activation function used to transform inputs
        """
        assert n_nodes >= n_inputs
        assert len(w) == n_nodes + n_outputs - n_inputs # first index of w
        assert len(b) == n_nodes + n_outputs - n_inputs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes
        self.w = w
        self.b = b
        self.activation = activation

    def set_keras_sequential(self,keras_model):
        """
            Unpack a keras model into dictionaries of weights and biases.  The dictionaries are used to build the underlying pyomo model.
        """
        chain = keras_model
        w,b = _sequential_to_dict(chain)
        self.w = w
        self.b = b
        self.n_inputs = len(chain.get_weights()[0])
        self.n_outputs = len(chain.get_weights()[-1])
        self.n_nodes = len(w) - self.n_outputs + self.n_inputs

    def build(self,mode = EncodedMode()):
        """
            Build the pyomo block using the given mode.
        """
        self.mode = mode
        self.mode.build(self)
